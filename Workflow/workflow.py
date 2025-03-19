import os
import numpy as np
import torch
import pandas as pd
import argparse
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from huggingface_mae import MAEModel

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
    This returns the filename without the ".npz" extension.
    For example, if the file is "plateA-12345.npz", it returns "plateA-12345".
    """
    base = os.path.basename(filename)
    if base.endswith(".npz"):
        return base[:-4]
    return base

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
# Main workflow function
# ---------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CSV with sample keys, SMILES strings, and labels.
    # CSV is expected to have columns: "SAMPLE_KEY", "CPD_SMILES", and optionally "label".
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
    
    results = []
    
    # Process each NPZ file in the given folder (plate folder)
    for file in os.listdir(args.npz_folder):
        if file.endswith(".npz"):
            npz_path = os.path.join(args.npz_folder, file)
            sample_key = extract_sample_key(file)
            print(f"Processing sample: {sample_key}")
            
            # Retrieve the corresponding row from the CSV using the SAMPLE_KEY column.
            row = df[df["SAMPLE_KEY"] == sample_key]
            if row.empty:
                print(f"No matching CSV entry found for sample key: {sample_key}. Skipping.")
                continue
            
            smiles = row.iloc[0]["CPD_SMILES"]
            true_label = row.iloc[0]["label"] if "label" in row.columns else None
            
            # Generate embeddings
            img_embedding = get_openphenom_embedding(npz_path, openphenom_model, device)  # shape: (1, D_img)
            smiles_embedding = get_chemberta_embedding(smiles, chemberta_tokenizer, chemberta_model, device)  # shape: (1, D_smiles)
            
            # Forward pass through the projection and classifier network
            logits = net(img_embedding, smiles_embedding)  # shape: (1, 2)
            prediction = torch.argmax(logits, dim=1).item()
            pred_label = "active" if prediction == 1 else "inactive"
            print(f"Sample {sample_key}: Predicted {pred_label}")
            
            results.append({
                "SAMPLE_KEY": sample_key,
                "predicted_label": pred_label,
                "true_label": true_label
            })
    
    # Save predictions to CSV if an output path is provided.
    if args.output_csv:
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_csv, index=False)
        print(f"Predictions saved to {args.output_csv}")

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
                        help="Path to the FINAL_LABEL_DF.csv file with columns 'SAMPLE_KEY', 'CPD_SMILES', and optionally 'label'.")
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="File path to save the prediction results.")
    parser.add_argument("--proj_dim", type=int, default=256,
                        help="Dimension for the projection layers into the shared vector space.")
    parser.add_argument("--fcn_hidden_dim", type=int, default=128,
                        help="Hidden dimension for the fully connected classifier network.")
    
    args = parser.parse_args()
    main(args)


# Example usage 
# python workflow.py --npz_folder /home/s2639050/MLP/final_image_path --csv_file /home/s2639050/MLP/FINAL_LABEL_DF.csv --output_csv /home/s2639050/MLP/predictions.csv