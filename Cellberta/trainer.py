import os
from typing import Union
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.tracking import WandBTracker
from datasets import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import wandb

from cellberta.configs import Configs
from cellberta.datasets.utils import DataCollatorWithPadding, prepare_dataset_from_dataframe
from cellberta.models import CellBERTa
from cellberta.models.utils import load_trained_model, load_pretrained_activity_bounds
from cellberta.metrics import get_ci, get_pearson, get_rmse, get_spearman
import numpy as np


class Trainer:
    """
    The Trainer class handles the training, validation, and testing processes for the CellBERTa models.
    It supports setting up datasets, initializing models, and managing the training loop with
    early stopping and learning rate scheduling.

    Attributes:
        configs (Configs): Configuration object with all necessary hyperparameters and settings.
        wandb_entity (str): Weights & Biases entity name.
        wandb_project (str): Weights & Biases project name.
        outputs_dir (str): Directory where output files such as checkpoints and logs are saved.
    """

    def __init__(
        self, configs: Configs, wandb_entity: str, wandb_project: str, outputs_dir: str
    ):
        """
        Initialize the Trainer with the provided configurations, Weights & Biases settings, 
        and output directory.

        Args:
            configs (Configs): Configuration object.
            wandb_entity (str): Weights & Biases entity name.
            wandb_project (str): Weights & Biases project name.
            outputs_dir (str): Directory where outputs are saved.
        """
        self.configs = configs

        self.dataset_configs = self.configs.dataset_configs
        self.training_configs = self.configs.training_configs
        self.model_configs = self.configs.model_configs

        self.gradient_accumulation_steps = (
            self.model_configs.model_hyperparameters.gradient_accumulation_steps
        )
        self.image_max_size = (
            self.model_configs.model_hyperparameters.image_max_size
        )
        self.drug_max_seq_len = (
            self.model_configs.model_hyperparameters.drug_max_seq_len
        )

        self.outputs_dir = outputs_dir
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project

        # Load the drug tokenizer for SMILES strings
        self.drug_tokenizer = self._load_tokenizer()

        # Initialize the model
        self.model = CellBERTa(self.model_configs)

        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None

        self._setup_run_name()

    def _load_tokenizer(self):
        """
        Load the tokenizer for drug SMILES.

        Returns:
            AutoTokenizer: The drug tokenizer.
        """
        return AutoTokenizer.from_pretrained(self.model_configs.drug_model_name_or_path)

    def set_activity_bounds(self, dataset):
        """
        Set the activity bounds for scaling the labels in the dataset. If a checkpoint is loaded 
        for a zero-shot experiment, the bounds are loaded from the checkpoint.

        Args:
            dataset: The dataset containing the activity labels.
        """
        # For a new dataset, determine bounds from the data
        self.activity_lower_bound = min(dataset.y)
        self.activity_upper_bound = max(dataset.y)

        # Load activity bounds from a trained model if performing a zero-shot experiment
        if self.model_configs.checkpoint_path:
            if self.dataset_configs.train_ratio == 0.0:
                self.activity_lower_bound, self.activity_upper_bound = load_pretrained_activity_bounds(
                    self.model_configs.checkpoint_path
                )
        
        print(
            f"Scaling labels: from {self.activity_lower_bound} - {self.activity_upper_bound} to -1 to 1"
        )

    def set_dataset(self, data_df=None, image_base_path=None, smiles_col="SMILES", activity_col="Activity", train_ratio=None):
        """
        Prepare and set up the dataset for training, validation, and testing.

        Args:
            data_df (pd.DataFrame, optional): DataFrame containing the dataset.
            image_base_path (str, optional): Base path to the image files.
            smiles_col (str, optional): Column name for SMILES strings.
            activity_col (str, optional): Column name for activity values.
            train_ratio (float, optional): Ratio of data to use for training.

        Returns:
            dict: Dictionary containing the dataset splits (train, valid, test).
        """
        # If no data_df provided, load from the configured dataset name
        if data_df is None:
            if self.dataset_configs.dataset_name:
                # Load from a predefined dataset name (e.g., from HuggingFace)
                try:
                    from datasets import load_dataset
                    data_df = load_dataset("PATH_TO_YOUR_DATASET", self.dataset_configs.dataset_name, split="train").to_pandas()
                except:
                    raise ValueError(f"Could not load dataset: {self.dataset_configs.dataset_name}")
            else:
                raise ValueError("Either provide a DataFrame or specify a dataset name in configs")
        
        # If train_ratio is provided, override the config
        if train_ratio is not None:
            self.dataset_configs.train_ratio = train_ratio
            
        # Extract activity values from the dataset
        y_values = data_df[activity_col].values
            
        print(f"Training with {self.model_configs.loss_function} loss function.")

        # Apply activity scaling if using cosine MSE loss
        if self.model_configs.loss_function == "cosine_mse":
            self.set_activity_bounds(pd.Series(y_values))

            # Scale activity values to [-1, 1] range
            if self.activity_upper_bound == self.activity_lower_bound:
                # Handle case where all labels are the same
                data_df["scaled_activity"] = 0
            else:
                data_df["scaled_activity"] = data_df[activity_col].apply(
                    lambda x: (x - self.activity_lower_bound) / 
                    (self.activity_upper_bound - self.activity_lower_bound) * 2 - 1
                )
            
            # Use the scaled activity for training
            activity_col = "scaled_activity"

        # Split the dataset into train/valid/test
        dataset_splits = {}
        
        # For zero-shot or few-shot learning
        if self.dataset_configs.train_ratio is not None:
            # Shuffle the data
            data_df = data_df.sample(frac=1, random_state=self.training_configs.random_seed).reset_index(drop=True)
            
            # Determine validation and test split sizes
            val_size = int(0.1 * len(data_df))  # 10% for validation
            
            if self.dataset_configs.train_ratio > 0:
                # Use a portion for training
                train_size = int(self.dataset_configs.train_ratio * (len(data_df) - val_size))
                
                # Split the data
                valid_df = data_df[:val_size]
                remaining_df = data_df[val_size:]
                train_df = remaining_df[:train_size]
                test_df = remaining_df[train_size:]
                
                dataset_splits["train"] = train_df
                dataset_splits["valid"] = valid_df
                dataset_splits["test"] = test_df
            else:
                # For zero-shot, only use test data
                dataset_splits["train"] = None
                dataset_splits["valid"] = None
                dataset_splits["test"] = data_df
        else:
            # For standard training, use the split_method from configs
            raise NotImplementedError("Custom split methods not implemented yet")

        # Create DataLoaders for each split
        if "train" in dataset_splits and dataset_splits["train"] is not None:
            print(f"Setup Train DataLoader")
            train_dataset = prepare_dataset_from_dataframe(
                dataset_splits["train"], 
                self.drug_tokenizer, 
                image_base_path, 
                activity_col
            )
            self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
            
        if "valid" in dataset_splits and dataset_splits["valid"] is not None:
            print(f"Setup Valid DataLoader")
            valid_dataset = prepare_dataset_from_dataframe(
                dataset_splits["valid"], 
                self.drug_tokenizer, 
                image_base_path, 
                activity_col
            )
            self.valid_dataloader = self._create_dataloader(valid_dataset, shuffle=False)
            
        print(f"Setup Test DataLoader")
        test_dataset = prepare_dataset_from_dataframe(
            dataset_splits["test"], 
            self.drug_tokenizer, 
            image_base_path, 
            activity_col
        )
        self.test_dataloader = self._create_dataloader(test_dataset, shuffle=False)
        
        return dataset_splits

    def _create_dataloader(self, dataset, shuffle=False):
        """
        Create a DataLoader from a dataset.
        
        Args:
            dataset: The dataset to create a DataLoader from.
            shuffle (bool): Whether to shuffle the data.
            
        Returns:
            DataLoader: The created DataLoader.
        """
        data_collator = DataCollatorWithPadding(
            drug_tokenizer=self.drug_tokenizer,
            padding="max_length",
            drug_max_length=self.drug_max_seq_len,
            return_tensors="pt",
        )
        
        return DataLoader(
            dataset,
            shuffle=shuffle,
            collate_fn=data_collator,
            batch_size=self.training_configs.batch_size,
            pin_memory=True,
        )

    def _setup_run_name(self):
        """
        Setup the run name and group name for the Weights & Biases tracker based on
        the dataset, split method, and model hyperparameters.
        """
        image_peft_hyperparameters = (
            self.model_configs.image_peft_hyperparameters
        )
        drug_peft_hyperparameters = self.model_configs.drug_peft_hyperparameters

        # Group name depends on the dataset and split method
        self.group_name = f"{self.dataset_configs.dataset_name}_{self.dataset_configs.split_method}"

        # Run name depends on the fine-tuning type and other relevant hyperparameters
        hyperparams = []
        hyperparams += [f"image_{self.model_configs.image_fine_tuning_type}"]
        if image_peft_hyperparameters:
            for key, value in image_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        hyperparams += [f"drug_{self.model_configs.drug_fine_tuning_type}"]
        if drug_peft_hyperparameters:
            for key, value in drug_peft_hyperparameters.items():
                if key not in ["target_modules", "feedforward_modules"]:
                    hyperparams += [f"{key}_{value}"]
        self.run_name = "_".join(hyperparams)

    def setup_training(self):
        """
        Setup the training environment, including initializing the Accelerator, WandB tracker, 
        optimizer, and learning rate scheduler. Prepares the model and dataloaders for training.
        """
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            log_with="wandb",
        )
        self.wandb_tracker = None
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                init_kwargs={
                    "wandb": {
                        "entity": self.wandb_entity,
                        "name": self.run_name,
                        "group": self.group_name,
                    }
                },
                config=self.configs.dict(),
            )
            self.wandb_tracker: WandBTracker = self.accelerator.get_tracker("wandb")
        self.accelerator.wait_for_everyone()

        if self.train_dataloader is not None:
            # Initialize optimizer with parameters that require gradients
            self.optimizer = AdamW(
                params=[
                    param
                    for name, param in self.model.named_parameters()
                    if param.requires_grad
                    and "noise_sigma" not in name  # Handle Balanced MSE loss
                ],
                lr=self.model_configs.model_hyperparameters.learning_rate,
            )

            # Setup learning rate scheduler
            num_training_steps = (
                len(self.train_dataloader) * self.training_configs.epochs
            )
            warmup_steps_ratio = (
                self.model_configs.model_hyperparameters.warmup_steps_ratio
            )
            self.lr_scheduler = get_linear_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=warmup_steps_ratio,
                num_training_steps=num_training_steps,
            )

            # Prepare model, dataloaders, optimizer, and scheduler for training
            (
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            ) = self.accelerator.prepare(
                self.model,
                self.train_dataloader,
                self.valid_dataloader,
                self.test_dataloader,
                self.optimizer,
                self.lr_scheduler,
            )
        else:
            # If only testing, prepare the model and test dataloader
            (
                self.model,
                self.test_dataloader,
            ) = self.accelerator.prepare(
                self.model,
                self.test_dataloader,
            )

        # Load a trained model from checkpoint if specified
        if self.model_configs.checkpoint_path:
            load_trained_model(
                self.model, 
                self.model_configs, 
                is_training=self.train_dataloader is not None
            )

    def compute_metrics(self, labels, predictions):
        """
        Compute evaluation metrics including RMSE, Pearson, Spearman, and CI.

        Args:
            labels (Tensor): True labels.
            predictions (Tensor): Predicted values.

        Returns:
            dict: Dictionary containing the computed metrics.
        """
        if self.model_configs.loss_function in ["cosine_mse"]:
            # Rescale predictions and labels back to the original activity range
            activity_range = self.activity_upper_bound - self.activity_lower_bound
            labels = (labels + 1) / 2 * activity_range + self.activity_lower_bound
            predictions = (predictions + 1) / 2 * activity_range + self.activity_lower_bound

        rmse = get_rmse(labels, predictions)
        pearson = get_pearson(labels, predictions)
        spearman = get_spearman(labels, predictions)
        ci = get_ci(labels, predictions)

        return {
            "rmse": rmse,
            "pearson": pearson,
            "spearman": spearman,
            "ci": ci,
        }

    def train(self):
        """
        Execute the training loop, handling early stopping, checkpoint saving, and logging metrics.
        """
        if self.train_dataloader is None:
            epoch = 0
            best_checkpoint_dir = None
        else:
            best_loss = 999999999
            patience = self.training_configs.patience
            eval_train_every_n_epochs = max(1, self.training_configs.epochs // 4)
            epochs_no_improve = 0  # Initialize early stopping counter

            print("Trainable params")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(name)

            for epoch in range(self.training_configs.epochs):
                self.model.train()

                num_train_steps = len(self.train_dataloader)
                progress_bar = tqdm(
                    total=int(num_train_steps // self.gradient_accumulation_steps),
                    position=0,
                    leave=True,
                    disable=not self.accelerator.is_local_main_process,
                )
                total_train_loss = 0
                for train_step, batch in enumerate(self.train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        outputs = self.model(batch)
                        loss = outputs["loss"]

                        # Backpropagation
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.model.zero_grad()
                        self.optimizer.zero_grad()

                        progress_bar.set_description(f"Epoch {epoch}; Loss: {loss:.4f}")
                        total_train_loss += loss.detach().float()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if self.accelerator.sync_gradients:
                        progress_bar.update(1)

                if (epoch + 1) % eval_train_every_n_epochs == 0:
                    train_metrics = self.test("train")
                else:
                    train_metrics = {
                        "train/loss": total_train_loss / len(self.train_dataloader)
                    }
                # At the end of an epoch, compute validation metrics
                valid_metrics = self.test("valid")

                if valid_metrics:
                    current_loss = valid_metrics["valid/loss"]
                else:
                    # Just train until the last epoch
                    current_loss = best_loss
                if current_loss <= best_loss:
                    best_loss = current_loss
                    epochs_no_improve = 0
                    # Save the model
                    best_checkpoint_dir = f"step_{epoch}"
                    self.accelerator.save_state(
                        os.path.join(
                            self.outputs_dir, "checkpoint", best_checkpoint_dir
                        )
                    )
                else:
                    epochs_no_improve += 1

                self.accelerator.log(train_metrics | valid_metrics, step=epoch)

                if epochs_no_improve >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Reload the best model checkpoint
            if best_checkpoint_dir:
                self.accelerator.load_state(
                    os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
                )
                self.accelerator.wait_for_everyone()

        # Compute test metrics and log results
        test_metrics = self.test("test", save_prediction=True)
        self.accelerator.log(test_metrics, step=epoch)

        # Save train and validation predictions if available
        if self.train_dataloader is not None:
            train_metrics = self.test("train", save_prediction=True)
            self.accelerator.log(train_metrics, step=epoch)
        
        if self.valid_dataloader is not None:
            valid_metrics = self.test("valid", save_prediction=True)
            self.accelerator.log(valid_metrics, step=epoch)

        if best_checkpoint_dir:
            print("Create a WandB artifact from embedding")
            artifact = wandb.Artifact(best_checkpoint_dir, type="model")
            artifact.add_dir(
                os.path.join(self.outputs_dir, "checkpoint", best_checkpoint_dir)
            )
            wandb.log_artifact(artifact)

    def test(self, split: str, save_prediction=False):
        """
        Evaluate the model on the specified dataset split and optionally save predictions.

        Args:
            split (str): The dataset split to evaluate on ('train', 'valid', 'test').
            save_prediction (bool): Whether to save the predictions as a CSV file.

        Returns:
            dict: Dictionary containing the evaluation metrics for the specified split.
        """
        if split == "train":
            dataloader = self.train_dataloader
        elif split == "valid":
            dataloader = self.valid_dataloader
        elif split == "test":
            dataloader = self.test_dataloader

        if dataloader is None:
            return {}

        total_loss = 0
        all_images = []
        all_drugs = []
        all_labels = []
        all_predictions = []

        self.model.eval()

        num_steps = len(dataloader)
        progress_bar = tqdm(
            total=num_steps,
            position=0,
            leave=True,
            disable=not self.accelerator.is_local_main_process,
        )
        for step, batch in enumerate(tqdm(dataloader)):
            with torch.no_grad():
                outputs = self.model(batch)
                loss = outputs["loss"]
                total_loss += loss.detach().float()

                # Collect predictions and labels for metric computation
                if "image_path" in batch:
                    all_images += batch["image_path"]
                all_drugs += batch["drug_ori_sequences"]
                if self.model_configs.loss_function == "cosine_mse":
                    all_labels += [batch["labels"]]
                    all_predictions += [outputs["cosine_similarity"]]
                elif self.model_configs.loss_function == "baseline_mse":
                    all_labels += [batch["labels"]]
                    all_predictions += [outputs["logits"]]

            progress_bar.set_description(f"Eval: {split} split")
            progress_bar.update(1)

        # Concatenate all predictions and labels across batches
        all_labels = torch.cat(all_labels, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        performance_metrics = self.compute_metrics(all_labels, all_predictions)
        metrics = {
            f"{split}/loss": total_loss / len(dataloader),
        }
        for metric_name, metric_value in performance_metrics.items():
            metrics[f"{split}/{metric_name}"] = metric_value

        if save_prediction:
            # Save predictions and labels to a CSV file
            df = pd.DataFrame(columns=["image", "drug", "label", "prediction"])
            if all_images:
                df["image"] = all_images
            df["drug"] = all_drugs

            if self.model_configs.loss_function in [
                "cosine_mse"
            ]:
                activity_range = self.activity_upper_bound - self.activity_lower_bound
                all_labels = (all_labels + 1) / 2 * activity_range + self.activity_lower_bound
                all_predictions = (
                    all_predictions + 1
                ) / 2 * activity_range + self.activity_lower_bound

            df["label"] = all_labels.cpu().numpy().tolist()
            df["prediction"] = all_predictions.cpu().numpy().tolist()
            df.to_csv(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))

            # Log the predictions as a WandB artifact
            artifact = wandb.Artifact(f"{split}_prediction", type="prediction")
            artifact.add_file(os.path.join(self.outputs_dir, f"{split}_prediction.csv"))
            wandb.log_artifact(artifact)

        return metrics