import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class OpenPHENOMChemBERTAConnector(nn.Module):
    """
    Connector model that bridges OpenPHENOM vision transformer with ChemBERTA embeddings
    using projection layers and cosine similarity for classification.
    """
    def __init__(
        self,
        vision_model_name="instafilter/OpenPhenomenalB",
        chemical_model_name="DeepChem/ChemBERTa-77M-MTR",
        projection_dim=512,
        n_classes=2,
        dropout_rate=0.1,
        combination_method="cosine"  # Options: cosine, concat, sum, subtract
    ):
        super(OpenPHENOMChemBERTAConnector, self).__init__()
        
        # Load vision transformer (OpenPHENOM)
        self.vision_model = AutoModelForImageClassification.from_pretrained(vision_model_name)
        self.vision_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        
        # Load ChemBERTA model
        self.chem_model = AutoModel.from_pretrained(chemical_model_name)
        self.chem_tokenizer = AutoTokenizer.from_pretrained(chemical_model_name)
        
        # Get embedding dimensions
        self.vision_dim = self.vision_model.classifier.in_features
        self.chem_dim = self.chem_model.config.hidden_size
        
        # Remove the classification head from the vision model
        self.vision_model.classifier = nn.Identity()
        
        # Projection layers
        self.vision_projection = nn.Sequential(
            nn.Linear(self.vision_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        self.chem_projection = nn.Sequential(
            nn.Linear(self.chem_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU()
        )
        
        self.combination_method = combination_method
        
        # Classification head depends on the combination method
        if combination_method == "cosine":
            self.classifier = nn.Linear(1, n_classes)
        elif combination_method == "concat":
            self.classifier = nn.Linear(projection_dim * 2, n_classes)
        else:  # sum or subtract
            self.classifier = nn.Linear(projection_dim, n_classes)
            
    def get_embeddings(self, images, smiles_strings):
        """
        Extract and project embeddings from both modalities.
        
        Args:
            images: Batch of images
            smiles_strings: Batch of SMILES strings
            
        Returns:
            Tuple: (vision_projected, chem_projected)
        """
        # Process images and extract features
        vision_inputs = self.vision_processor(images, return_tensors="pt").to(images.device)
        vision_outputs = self.vision_model(**vision_inputs)
        vision_embeddings = vision_outputs.logits  # Using the features before classification
        
        # Process SMILES strings and extract features
        chem_inputs = self.chem_tokenizer(smiles_strings, padding=True, truncation=True, 
                                         return_tensors="pt").to(images.device)
        chem_outputs = self.chem_model(**chem_inputs)
        chem_embeddings = chem_outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
        
        # Project embeddings to the same dimension
        vision_projected = self.vision_projection(vision_embeddings)
        chem_projected = self.chem_projection(chem_embeddings)
        
        return vision_projected, chem_projected
        
    def forward(self, images, smiles_strings):
        """
        Forward pass through the multimodal connector
        
        Args:
            images: Batch of images
            smiles_strings: Batch of SMILES strings
        
        Returns:
            dict: Contains 'logits' (classification logits) and additional information 
                 based on the combination method
        """
        # Get projected embeddings
        vision_projected, chem_projected = self.get_embeddings(images, smiles_strings)
        
        # Combine embeddings based on the specified method
        if self.combination_method == "cosine":
            # Normalize for cosine similarity
            vision_norm = F.normalize(vision_projected, p=2, dim=1)
            chem_norm = F.normalize(chem_projected, p=2, dim=1)
            
            # Calculate cosine similarity
            similarity = torch.sum(vision_norm * chem_norm, dim=1, keepdim=True)
            combined = similarity
            
        elif self.combination_method == "concat":
            combined = torch.cat((vision_projected, chem_projected), dim=1)
            
        elif self.combination_method == "sum":
            combined = vision_projected + chem_projected
            
        elif self.combination_method == "subtract":
            combined = vision_projected - chem_projected
        
        # Classification
        logits = self.classifier(combined)
        
        # Return a dictionary with results (similar to the BALM model in the provided code)
        results = {
            "logits": logits,
            "vision_embedding": vision_projected,
            "chem_embedding": chem_projected
        }
        
        # Add cosine similarity if it was calculated
        if self.combination_method == "cosine":
            results["cosine_similarity"] = similarity
            
        return results


def argument_parser():
    """Parse command line arguments for the connector script."""
    parser = argparse.ArgumentParser(description="Train an OpenPHENOM+ChemBERTA connector model.")
    
    parser.add_argument("--embedding_method", type=str, default="cosine",
                      choices=["cosine", "concat", "sum", "subtract"],
                      help="Method to combine embeddings")
    
    parser.add_argument("--projection_dim", type=int, default=512,
                      help="Dimension of the projection layers")
    
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate for training")
    
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for training")
    
    parser.add_argument("--epochs", type=int, default=10,
                      help="Number of training epochs")
    
    parser.add_argument("--test_size", type=float, default=0.2,
                      help="Proportion of data to use for testing")
    
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Random seed for reproducibility")
    
    return parser.parse_args()


def train_and_evaluate(model, train_dataloader, test_dataloader, args):
    """
    Train and evaluate the OpenPHENOM+ChemBERTA connector model.
    
    Args:
        model: The connector model
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for test data
        args: Command line arguments
        
    Returns:
        dict: Evaluation metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch in train_dataloader:
            images, smiles, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, smiles)
            logits = outputs["logits"]
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            images, smiles, labels = batch
            images = images.to(device)
            
            outputs = model(images, smiles)
            logits = outputs["logits"]
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    print(f"Test Results: Accuracy={accuracy:.4f}, F1={f1:.4f}")
    return metrics


def load_and_preprocess_data(image_paths, smiles_list, labels, args):
    """
    Load and preprocess the multimodal data.
    
    Args:
        image_paths: List of paths to images
        smiles_list: List of SMILES strings
        labels: List of class labels
        args: Command line arguments
        
    Returns:
        tuple: Train and test dataloaders
    """
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image
    
    class MultimodalDataset(Dataset):
        def __init__(self, image_paths, smiles_list, labels, transform=None):
            self.image_paths = image_paths
            self.smiles_list = smiles_list
            self.labels = labels
            self.transform = transform
            
        def __len__(self):
            return len(self.image_paths)
            
        def __getitem__(self, idx):
            image = Image.open(self.image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            smiles = self.smiles_list[idx]
            label = self.labels[idx]
            
            return image, smiles, label
    
    # Split data into train and test sets
    img_train, img_test, smiles_train, smiles_test, labels_train, labels_test = train_test_split(
        image_paths, smiles_list, labels, test_size=args.test_size, random_state=args.random_seed
    )
    
    # Create datasets and dataloaders
    train_dataset = MultimodalDataset(img_train, smiles_train, labels_train)
    test_dataset = MultimodalDataset(img_test, smiles_test, labels_test)
    
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )
    
    return train_dataloader, test_dataloader


def main():
    """Main function to run the connector model."""
    args = argument_parser()
    
    # Initialize model
    model = OpenPHENOMChemBERTAConnector(
        projection_dim=args.projection_dim,
        combination_method=args.embedding_method
    )
    
    # Here you would load your dataset
    # For example:
    # image_paths, smiles_list, labels = load_your_dataset()
    # train_dataloader, test_dataloader = load_and_preprocess_data(
    #     image_paths, smiles_list, labels, args
    # )
    # metrics = train_and_evaluate(model, train_dataloader, test_dataloader, args)
    
    print("Model initialized successfully")
    print(f"Using {args.embedding_method} method for combining embeddings")
    print(f"Projection dimension: {args.projection_dim}")
    

if __name__ == "__main__":
    main()