import os
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# Define paths
csv_file_path = "D:/UoE/Coursework/MLP/Project/Labels_dataset/output/merged_cp_smiles.csv"
output_folder = "D:/UoE/Coursework/MLP/Project/Chemberta_embeddings_eachfile/"  # Output folder

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load CSV and extract SMILES
df = pd.read_csv(csv_file_path)
smiles_list = df["CPD_SMILES"].tolist()
file_names = df["File Name"].tolist()  # Use File Name column

# Process file names to remove ".npz"
processed_filenames = [fn.replace(".npz", ".npy") for fn in file_names]

# Load ChemBERTa tokenizer and model
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to process SMILES into embeddings
def get_chemberta_embeddings(smiles_list, batch_size=32):
    model.eval()
    
    with torch.no_grad():
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i + batch_size]
            batch_filenames = processed_filenames[i:i + batch_size]  # Get corresponding filenames
            
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state  # Get hidden states
            embeddings_batch = hidden_states.mean(dim=1).cpu().numpy()  # Mean pooling

            # Save each embedding as a separate .npy file
            for j, filename in enumerate(batch_filenames):
                embedding_file = os.path.join(output_folder, filename)
                np.save(embedding_file, embeddings_batch[j])

                print(f"Saved: {embedding_file}")

# Generate embeddings
print("Generating ChemBERTa embeddings...")
get_chemberta_embeddings(smiles_list, batch_size=32)

print(f"All embeddings saved in: {output_folder}")
