import argparse
import os
import sys
import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer
from huggingface_mae import MAEModel
from tqdm import tqdm

sys.path.append(os.getcwd())

from cellberta import common_utils
from cellberta.configs import Configs
from cellberta.models import CellBERTa
from cellberta.datasets.utils import CellImageProcessor
from cellberta.models.utils import load_trained_model
from cellberta.metrics import get_rmse, get_pearson, get_spearman, get_ci


def argument_parser():
    """
    Parse command-line arguments for the testing script.
    """
    parser = argparse.ArgumentParser(description="CellBERTa Testing")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the config file")
    parser.add_argument("--data_filepath", type=str, required=True, help="Path to the test data CSV file")
    parser.add_argument("--image_basepath", type=str, required=True, help="Base path to the image files")
    parser.add_argument("--activity_lower_bound", type=float, default=0.0, help="Lower bound for activity scaling")
    parser.add_argument("--activity_upper_bound", type=float, default=1.0, help="Upper bound for activity scaling")
    parser.add_argument("--checkpoint_path", type=str, help="Path to model checkpoint (override config)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for testing")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="Output file for predictions")
    args = parser.parse_args()
    return args


def main():
    """
    Main function for testing a CellBERTa model on new data.
    """
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))
    
    # Override checkpoint path if provided
    if args.checkpoint_path:
        configs.model_configs.checkpoint_path = args.checkpoint_path
    
    # Ensure checkpoint path is set
    if not configs.model_configs.checkpoint_path:
        raise ValueError("No checkpoint path specified. Use --checkpoint_path or set in config.")
    
    # Load the test data
    test_df = pd.read_csv(args.data_filepath)
    print(f"Loaded test data with {len(test_df)} samples")
    
    # Initialize model and tokenizer
    drug_tokenizer = AutoTokenizer.from_pretrained(configs.model_configs.drug_model_name_or_path)
    model = CellBERTa(configs.model_configs)
    model = load_trained_model(model, configs.model_configs, is_training=False)
    model.eval()
    
    # Create image processor
    image_processor = CellImageProcessor()
    
    # Process data and make predictions
    all_predictions = []
    all_labels = []
    all_image_paths = []
    all_smiles = []
    
    # Move model to appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"Making predictions on {len(test_df)} samples...")
    # Process in batches
    batch_size = args.batch_size
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_df = test_df.iloc[i:i+batch_size]
        
        # Process images
        image_tensors = []
        for _, row in batch_df.iterrows():
            image_path = os.path.join(args.image_basepath, f"{row['SampleKey']}.npz")
            all_image_paths.append(image_path)
            image_tensor = image_processor.load_npz_image(image_path)
            image_tensors.append(image_tensor)
        
        image_batch = torch.stack(image_tensors).to(device)
        
        # Process SMILES
        smiles_list = batch_df["SMILES"].tolist()
        all_smiles.extend(smiles_list)
        drug_inputs = drug_tokenizer(smiles_list, return_tensors="pt", padding=True, truncation=True).to(device)
        
        # Create input batch
        batch_input = {
            "image_tensor": image_batch,
            "drug_input_ids": drug_inputs["input_ids"],
            "drug_attention_mask": drug_inputs["attention_mask"],
        }
        
        # Make predictions
        with torch.no_grad():
            outputs = model(batch_input)
            predictions = outputs["cosine_similarity"]
            
            # Scale predictions from cosine similarity to activity values
            if configs.model_configs.loss_function == "cosine_mse":
                activity_range = args.activity_upper_bound - args.activity_lower_bound
                predictions = (predictions + 1) / 2 * activity_range + args.activity_lower_bound
        
        all_predictions.extend(predictions.cpu().numpy().tolist())
        
        # Get labels if available
        if "Activity" in batch_df.columns:
            all_labels.extend(batch_df["Activity"].tolist())
    
    # Create DataFrame with predictions
    results_df = pd.DataFrame({
        "image_path": all_image_paths,
        "SMILES": all_smiles,
        "prediction": all_predictions
    })
    
    # Add labels if available
    if all_labels:
        results_df["actual"] = all_labels
        
        # Calculate metrics
        predictions_tensor = torch.tensor(all_predictions)
        labels_tensor = torch.tensor(all_labels)
        
        rmse = get_rmse(labels_tensor, predictions_tensor).item()
        pearson = get_pearson(labels_tensor, predictions_tensor).item()
        spearman = get_spearman(labels_tensor, predictions_tensor)
        ci = get_ci(labels_tensor, predictions_tensor).item()
        
        print(f"RMSE: {rmse:.4f}")
        print(f"Pearson correlation: {pearson:.4f}")
        print(f"Spearman correlation: {spearman:.4f}")
        print(f"Concordance Index: {ci:.4f}")
    
    # Save predictions
    results_df.to_csv(args.output_file, index=False)
    print(f"Saved predictions to {args.output_file}")


if __name__ == "__main__":
    main()