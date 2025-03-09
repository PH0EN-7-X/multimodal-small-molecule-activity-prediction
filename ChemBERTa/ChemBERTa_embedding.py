import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Path to the input CSV file
csv_file_path = "D:/UoE/Coursework/MLP/Project/Labels_dataset/output/merged_cp_smiles.csv"  # Your uploaded file

# Load CSV and extract SMILES
df = pd.read_csv(csv_file_path)
smiles_list = df["CPD_SMILES"].tolist()  # Adjust column name if needed

# Load ChemBERTa tokenizer and model
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to process SMILES into embeddings
def get_chemberta_embeddings(smiles_list, batch_size=32):
    embeddings = []
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for efficiency
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # Get hidden states
            embeddings_batch = hidden_states.mean(dim=1).cpu().numpy()  # Mean-pooling
            # embeddings_batch = hidden_states[:, 0, :].cpu().numpy()  # CLS-pooling
            embeddings.append(embeddings_batch)

    return np.vstack(embeddings)  # Stack all embeddings

# Generate embeddings
print("Generating ChemBERTa embeddings...")
smiles_embeddings = get_chemberta_embeddings(smiles_list, batch_size=32)

# Save embeddings as a NumPy file
output_file_path = "D:/UoE/Coursework/MLP/Project/chemberta_smiles_embeddings_imgFiles.npy"
np.save(output_file_path, smiles_embeddings)

print(f"Saved embeddings to {output_file_path}")
