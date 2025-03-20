import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
from PIL import Image


class CellImageProcessor:
    """
    Utility class for processing cell images stored as NPZ files for the model.
    """
    
    @staticmethod
    def load_npz_image(npz_path):
        """
        Load and preprocess a NPZ file containing cell images.
        
        Args:
            npz_path (str): Path to the NPZ file
            
        Returns:
            torch.Tensor: Processed image tensor ready for the model
        """
        data = np.load(npz_path, allow_pickle=True)
        img_array = data['sample']
        data.close()
        
        # Transpose (H, W, 5) -> (5, H, W)
        img_array = np.transpose(img_array, (2, 0, 1))
        
        # Convert to float and normalize to [0,1]
        img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
        
        # Resize to (256, 256) if necessary
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension -> (1, 5, H, W)
        img_tensor = torch.nn.functional.interpolate(img_tensor, size=(256, 256), mode='bilinear', align_corners=False)
        img_tensor = img_tensor.squeeze(0)  # Remove batch dimension -> (5, 256, 256)
        
        return img_tensor


class DataCollatorWithPadding:
    """
    Data collator with padding for batching images and drug tokenized inputs.
    
    Args:
        drug_tokenizer: The tokenizer for drug SMILES.
        padding (str): Padding strategy ("max_length" or "longest").
        drug_max_length (int): Maximum sequence length for drug tokens.
        return_tensors (str): Format of the returned tensors (e.g., "pt" for PyTorch).
    """
    
    def __init__(
        self,
        drug_tokenizer,
        padding="longest",
        drug_max_length=None,
        return_tensors="pt",
    ):
        self.drug_tokenizer = drug_tokenizer
        self.padding = padding
        self.drug_max_length = drug_max_length
        self.return_tensors = return_tensors

    def __call__(self, examples):
        """
        Batch and pad examples.
        
        Args:
            examples (list): List of examples to batch.
            
        Returns:
            dict: Batched and padded examples.
        """
        batch = {}
        
        # Handle image data if present
        if "image_tensor" in examples[0]:
            batch["image_tensor"] = torch.stack([example["image_tensor"] for example in examples])
        
        # Handle drug tokenization data
        for key in ["drug_input_ids", "drug_attention_mask"]:
            if key in examples[0]:
                batch[key] = pad_sequence(
                    [example[key] for example in examples],
                    batch_first=True,
                    padding_value=self.drug_tokenizer.pad_token_id if "input_ids" in key else 0,
                )
                if self.padding == "max_length" and self.drug_max_length:
                    if batch[key].shape[1] > self.drug_max_length:
                        batch[key] = batch[key][:, :self.drug_max_length]
                    elif batch[key].shape[1] < self.drug_max_length:
                        padding_size = self.drug_max_length - batch[key].shape[1]
                        padding_value = self.drug_tokenizer.pad_token_id if "input_ids" in key else 0
                        padding_tensor = torch.ones(batch[key].shape[0], padding_size, dtype=batch[key].dtype) * padding_value
                        batch[key] = torch.cat([batch[key], padding_tensor], dim=1)
        
        # Add labels if present
        if "labels" in examples[0]:
            batch["labels"] = torch.tensor(
                [float(example["labels"]) for example in examples], dtype=torch.float32
            )
        
        # Add original sequences if present
        for key in ["drug_ori_sequences"]:
            if key in examples[0]:
                batch[key] = [example[key] for example in examples]
        
        # Add image paths if present
        if "image_path" in examples[0]:
            batch["image_path"] = [example["image_path"] for example in examples]
        
        return batch


class CellDrugDataset(Dataset):
    """
    Dataset class for cell images and drug SMILES pairs.
    
    Args:
        image_paths (list): List of paths to NPZ image files.
        smiles (list): List of drug SMILES strings.
        labels (list): List of activity labels.
        drug_tokenizer: Tokenizer for drug SMILES.
        transform (callable, optional): Optional transform to apply to images.
    """
    
    def __init__(self, image_paths, smiles, labels, drug_tokenizer, transform=None):
        self.image_paths = image_paths
        self.smiles = smiles
        self.labels = labels
        self.drug_tokenizer = drug_tokenizer
        self.transform = transform
        self.image_processor = CellImageProcessor()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and process image
        image_path = self.image_paths[idx]
        image_tensor = self.image_processor.load_npz_image(image_path)
        
        if self.transform:
            image_tensor = self.transform(image_tensor)
        
        # Tokenize SMILES
        drug_encoding = self.drug_tokenizer(
            self.smiles[idx],
            truncation=True,
            return_tensors="pt",
        )
        
        # Create item with all required data
        item = {
            "image_tensor": image_tensor,
            "drug_input_ids": drug_encoding["input_ids"].squeeze(0),
            "drug_attention_mask": drug_encoding["attention_mask"].squeeze(0),
            "drug_ori_sequences": self.smiles[idx],
            "image_path": image_path,
        }
        
        # Add label if present
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item


def prepare_dataset_from_dataframe(df, drug_tokenizer, image_base_path, activity_column="Activity"):
    """
    Create a CellDrugDataset from a DataFrame containing image paths and SMILES.
    
    Args:
        df (pd.DataFrame): DataFrame with columns for image keys, SMILES, and activity.
        drug_tokenizer: Tokenizer for drug SMILES.
        image_base_path (str): Base path where NPZ images are stored.
        activity_column (str): Name of the column containing activity values.
        
    Returns:
        CellDrugDataset: Dataset ready for training or evaluation.
    """
    # Extract data from DataFrame
    image_keys = df["SampleKey"].tolist()
    smiles = df["SMILES"].tolist()
    
    # Construct full image paths
    image_paths = [os.path.join(image_base_path, f"{key}.npz") for key in image_keys]
    
    # Extract labels if present
    labels = None
    if activity_column in df.columns:
        labels = df[activity_column].tolist()
    
    # Create and return dataset
    return CellDrugDataset(image_paths, smiles, labels, drug_tokenizer)