import os
import numpy as np
import torch
import pandas as pd
import argparse
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_mae import MAEModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

# ---------------------------
# Helper functions
# ---------------------------

def load_npz_image(npz_path):
    """
    Load and preprocess an NPZ image file.
    Assumes the NPZ file contains an array under key 'sample'
    and that the image shape is (H, W, 5).
    """
    data = np.load(npz_path, allow_pickle=True)
    img_array = data['sample']
    data.close()
    
    # Rearrange from (H, W, 5) to (5, H, W)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Convert to float and normalize to [0, 1]
    img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
    
    # Add batch dimension and resize to (256, 256)
    img_tensor = img_tensor.unsqueeze(0)  # shape: (1, 5, H, W)
    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
    
    return img_tensor  # Final shape: (1, 5, 256, 256)

def get_openphenom_embedding(npz_path, model, device):
    """
    Generate an image embedding using the OpenPhenom model.
    """
    input_tensor = load_npz_image(npz_path).to(device)
    with torch.no_grad():
        embedding = model.predict(input_tensor)
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding)
    return embedding  # Expected shape: (1, D_img)

def get_chemberta_embedding(smiles, tokenizer, model, device):
    """
    Generate an embedding for a SMILES string using ChemBERTa.
    """
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state  # shape: (1, seq_len, D_smiles)
    embedding = hidden_states.mean(dim=1)       # shape: (1, D_smiles)
    return embedding

def extract_sample_key(filename):
    """
    Extract the sample key from the filename.
    Returns the filename without the ".npz" extension.
    For example, "plateA-12345.npz" becomes "plateA-12345".
    """
    base = os.path.basename(filename)
    if base.endswith(".npz"):
        return base[:-4]
    return base

def label_to_int(label):
    """
    Convert label string to integer.
    Assumes "active" -> 1 and "inactive" -> 0.
    """
    label = str(label).strip().lower()
    return 1 if label == "active" else 0

# ---------------------------
# Define Projection and Classifier
# ---------------------------

class ProjectionClassifier(nn.Module):
    def __init__(self, img_embed_dim, smiles_embed_dim, proj_dim=256, fcn_hidden_dim=128, num_classes=2):
        """
        img_embed_dim: Dimension of OpenPhenom image embeddings.
        smiles_embed_dim: Dimension of ChemBERTa SMILES embeddings.
        proj_dim: Dimension to project each embedding into.
        fcn_hidden_dim: Hidden dimension for the FCN classifier.
        num_classes: Number of output classes (2 for binary classification).
        """
        super(ProjectionClassifier, self).__init__()
        self.img_proj = nn.Linear(img_embed_dim, proj_dim)
        self.smiles_proj = nn.Linear(smiles_embed_dim, proj_dim)
        self.classifier = nn.Sequential(
            nn.Linear(2 * proj_dim, fcn_hidden_dim),
            nn.ReLU(),
            nn.Linear(fcn_hidden_dim, num_classes)
        )
    
    def forward(self, img_embedding, smiles_embedding):
        proj_img = self.img_proj(img_embedding)
        proj_smiles = self.smiles_proj(smiles_embedding)
        concat = torch.cat([proj_img, proj_smiles], dim=1)
        out = self.classifier(concat)
        return out

# ---------------------------
# Training Function
# ---------------------------

def train_model(args, device, openphenom_model, chemberta_model, chemberta_tokenizer, net, df):
    net.train()  # Set projection/classifier to training mode
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Build a list of npz file paths with valid CSV labels.
    file_list = []
    for file in os.listdir(args.npz_folder):
        if file.endswith(".npz"):
            sample_key = extract_sample_key(file)
            if not df[df["SAMPLE_KEY"] == sample_key].empty:
                file_list.append(file)
    
    print(f"Found {len(file_list)} samples for training.")
    
    for epoch in range(args.epochs):
        epoch_losses = []
        for file in file_list:
            npz_path = os.path.join(args.npz_folder, file)
            sample_key = extract_sample_key(file)
            
            # Retrieve CSV row for this sample key.
            row = df[df["SAMPLE_KEY"] == sample_key]
            if row.empty:
                continue
            smiles = row.iloc[0]["CPD_SMILES"]
            true_label = label_to_int(row.iloc[0]["label"])
            
            # Generate embeddings
            img_embedding = get_openphenom_embedding(npz_path, openphenom_model, device)  # shape: (1, D_img)
            smiles_embedding = get_chemberta_embedding(smiles, chemberta_tokenizer, chemberta_model, device)  # shape: (1, D_smiles)
            
            # Forward pass through projection/classifier network
            logits = net(img_embedding, smiles_embedding)  # shape: (1, 2)
            loss = criterion(logits, torch.tensor([true_label], dtype=torch.long, device=device))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else float('nan')
        print(f"Epoch [{epoch+1}/{args.epochs}], Average Loss: {avg_loss:.4f}")
    
    # Optionally, save the trained network
    if args.save_model:
        torch.save(net.state_dict(), args.save_model)
        print(f"Trained model saved to {args.save_model}")

# ---------------------------
# Evaluation Function
# ---------------------------

def evaluate_model(args, device, openphenom_model, chemberta_model, chemberta_tokenizer, net, df):
    net.eval()  # Set network to evaluation mode
    results = []
    
    # Process each NPZ file in the folder
    for file in os.listdir(args.npz_folder):
        if file.endswith(".npz"):
            npz_path = os.path.join(args.npz_folder, file)
            sample_key = extract_sample_key(file)
            print(f"Processing sample: {sample_key}")
            
            # Retrieve CSV row for this sample.
            row = df[df["SAMPLE_KEY"] == sample_key]
            if row.empty:
                print(f"No matching CSV entry found for sample key: {sample_key}. Skipping.")
                continue
            
            smiles = row.iloc[0]["CPD_SMILES"]
            true_label = row.iloc[0]["label"] if "label" in row.columns else None
            
            # Generate embeddings
            img_embedding = get_openphenom_embedding(npz_path, openphenom_model, device)
            smiles_embedding = get_chemberta_embedding(smiles, chemberta_tokenizer, chemberta_model, device)
            
            # Forward pass through the network
            logits = net(img_embedding, smiles_embedding)
            prediction = torch.argmax(logits, dim=1).item()
            pred_label = "active" if prediction == 1 else "inactive"
            print(f"Sample {sample_key}: Predicted {pred_label}")
            
            results.append({
                "SAMPLE_KEY": sample_key,
                "predicted_label": pred_label,
                "true_label": true_label
            })
    
    # Save predictions if output CSV path is provided.
    if args.output_csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"Predictions saved to {args.output_csv}")
    
    # Calculate Evaluation Metrics if true labels are available.
    valid_results = [r for r in results if r["true_label"] is not None]
    if valid_results:
        label_map = {"inactive": 0, "active": 1}
        y_true = []
        y_pred = []
        for r in valid_results:
            true_val = str(r["true_label"]).strip().lower()
            pred_val = str(r["predicted_label"]).strip().lower()
            if true_val in label_map and pred_val in label_map:
                y_true.append(label_map[true_val])
                y_pred.append(label_map[pred_val])
        
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, pos_label=1)
            rec = recall_score(y_true, y_pred, pos_label=1)
            f1 = f1_score(y_true, y_pred, pos_label=1)
            iou = jaccard_score(y_true, y_pred, pos_label=1)
            
            print("\nEvaluation Metrics:")
            print(f"Accuracy:  {acc:.4f}")
            print(f"Precision: {prec:.4f}")
            print(f"Recall:    {rec:.4f}")
            print(f"F1 Score:  {f1:.4f}")
            print(f"IoU:       {iou:.4f}")
        else:
            print("No valid labels found for computing evaluation metrics.")
    else:
        print("No ground truth labels available; skipping metric calculations.")

# ---------------------------
# Main workflow function
# ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CSV with sample keys, SMILES strings, and labels.
    # CSV is expected to have columns: "SAMPLE_KEY", "CPD_SMILES", and "label".
    df = pd.read_csv(args.csv_file)
    
    # ---------------------------
    # Load pre-trained models
    # ---------------------------
    
    print("Loading OpenPhenom model...")
    openphenom_model = MAEModel.from_pretrained("recursionpharma/OpenPhenom")
    openphenom_model.eval()
    openphenom_model.to(device)
    
    print("Loading ChemBERTa model and tokenizer...")
    chemberta_model_name = "DeepChem/ChemBERTa-77M-MTR"
    chemberta_tokenizer = AutoTokenizer.from_pretrained(chemberta_model_name)
    chemberta_model = AutoModel.from_pretrained(chemberta_model_name)
    chemberta_model.eval()
    chemberta_model.to(device)
    
    # ---------------------------
    # Determine embedding dimensions using dummy inputs
    # ---------------------------
    
    dummy_npz = None
    for file in os.listdir(args.npz_folder):
        if file.endswith(".npz"):
            dummy_npz = os.path.join(args.npz_folder, file)
            break
    if dummy_npz is None:
        print("No NPZ files found in the provided folder.")
        return
    
    openphenom_dummy = get_openphenom_embedding(dummy_npz, openphenom_model, device)
    img_embed_dim = openphenom_dummy.shape[1]
    
    dummy_smiles = "CCO"  # Example SMILES (ethanol)
    chemberta_dummy = get_chemberta_embedding(dummy_smiles, chemberta_tokenizer, chemberta_model, device)
    smiles_embed_dim = chemberta_dummy.shape[1]
    
    print(f"Image embedding dimension: {img_embed_dim}, SMILES embedding dimension: {smiles_embed_dim}")
    
    # ---------------------------
    # Initialize projection and classifier network
    # ---------------------------
    
    net = ProjectionClassifier(
        img_embed_dim=img_embed_dim, 
        smiles_embed_dim=smiles_embed_dim,
        proj_dim=args.proj_dim,
        fcn_hidden_dim=args.fcn_hidden_dim,
        num_classes=2
    )
    net.to(device)
    
    # ---------------------------
    # Run Training or Evaluation based on mode
    # ---------------------------
    
    if args.mode == "train":
        print("Starting training...")
        train_model(args, device, openphenom_model, chemberta_model, chemberta_tokenizer, net, df)
    else:
        print("Starting evaluation...")
        evaluate_model(args, device, openphenom_model, chemberta_model, chemberta_tokenizer, net, df)

# ---------------------------
# Command-line interface
# ---------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Workflow for predicting small molecule activity from cell microscopy images and SMILES embeddings."
    )
    parser.add_argument("--npz_folder", type=str, required=True,
                        help="Path to the folder containing NPZ image files for one plate. The folder name should match the plate prefix in the filenames.")
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to the FINAL_LABEL_DF.csv file with columns 'SAMPLE_KEY', 'CPD_SMILES', and 'label'.")
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="File path to save the prediction results (used in evaluation mode).")
    parser.add_argument("--proj_dim", type=int, default=256,
                        help="Dimension for the projection layers into the shared vector space.")
    parser.add_argument("--fcn_hidden_dim", type=int, default=128,
                        help="Hidden dimension for the fully connected classifier network.")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="eval",
                        help="Mode to run: 'train' to train the classifier, 'eval' to run inference and evaluation.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (used in train mode).")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer (used in train mode).")
    parser.add_argument("--save_model", type=str, default="",
                        help="Path to save the trained model (if training). Leave empty to skip saving.")
    
    args = parser.parse_args()
    main(args)
