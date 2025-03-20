import numpy as np
import pandas as pd
from datasets import load_dataset

class BaseDataset:
    """
    Base class for datasets used in CellBERTa models.
    Provides common functionality for loading and organizing dataset.
    """
    
    def __init__(self, dataset_name, train_ratio=None):
        """
        Initialize the base dataset.
        
        Args:
            dataset_name (str): Name of the dataset to load
            train_ratio (float, optional): Ratio of data to use for training
        """
        # Try to load from huggingface dataset or local file
        try:
            self.data = load_dataset("PATH_TO_YOUR_DATASET", dataset_name, split="train").to_pandas()
        except:
            # If not available in HF, try to load from local file
            try:
                self.data = pd.read_csv(dataset_name)
            except:
                raise ValueError(f"Could not load dataset: {dataset_name}")
        
        # Extract the target values
        self.y = self.data["Activity"].values

        self.train_ratio = train_ratio

        print(f"Total of original data: {len(self.data)}")
        # Remove rows with missing data
        self.data = self.data.dropna(subset=["SMILES"]).reset_index(drop=True)
        print(f"Total after filtering: {len(self.data)}")

    def get_split(self, *args, **kwargs):
        """
        Create data splits based on train_ratio.

        Returns:
            dict: A dictionary of DataFrames for each split ('train', 'valid', 'test').
        """
        if self.train_ratio > 0:
            # Separating data randomly
            ids = np.arange(len(self.data))
            np.random.shuffle(ids)

            # Define the size of each split
            val_size = int(0.1 * len(ids))  # 10% for validation
            train_size = int(
                self.train_ratio * (len(ids) - val_size)
            )  # Remaining for train

            # Split the ids
            val_ids = ids[:val_size]
            remaining_ids = ids[val_size:]
            train_ids = remaining_ids[:train_size]
            test_ids = remaining_ids[train_size:]

            # Create the splits
            train = self.data.iloc[train_ids].reset_index(drop=True)
            valid = self.data.iloc[val_ids].reset_index(drop=True)
            test = self.data.iloc[test_ids].reset_index(drop=True)
        else:
            train = None
            valid = None
            test = self.data

        return {
            "train": train,
            "valid": valid,
            "test": test,
        }