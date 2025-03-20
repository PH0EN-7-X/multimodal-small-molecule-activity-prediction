import argparse
import os
import sys

sys.path.append(os.getcwd())

from dotenv import load_dotenv

load_dotenv(".env")

from cellberta import common_utils
from cellberta.configs import Configs
from cellberta.trainer import Trainer
import pandas as pd


def argument_parser():
    parser = argparse.ArgumentParser(description="CellBERTa Training")
    parser.add_argument("--config_filepath", type=str, required=True, help="Path to the config file")
    parser.add_argument("--data_filepath", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--image_basepath", type=str, help="Base path to the image files")
    parser.add_argument("--smiles_col", type=str, default="SMILES", help="Column name for SMILES")
    parser.add_argument("--activity_col", type=str, default="Activity", help="Column name for activity values")
    parser.add_argument("--image_key_col", type=str, default="SampleKey", help="Column name for image keys")
    args = parser.parse_args()
    return args


def main() -> None:
    args = argument_parser()
    configs = Configs(**common_utils.load_yaml(args.config_filepath))

    # Set the random seed for reproducibility
    common_utils.setup_random_seed(configs.training_configs.random_seed)
    
    # Setup the output directory for the experiment
    outputs_dir = common_utils.setup_experiment_folder(
        os.path.join(os.getcwd(), configs.training_configs.outputs_dir)
    )
    
    # Save the training configuration to the output directory
    common_utils.save_training_configs(configs, outputs_dir)

    # Login to Weights & Biases (WandB)
    wandb_entity = os.getenv("WANDB_ENTITY", "")
    wandb_project = os.getenv("WANDB_PROJECT_NAME", "")

    # Load data if provided
    df = None
    if args.data_filepath:
        df = pd.read_csv(args.data_filepath)
        print(f"Loaded dataset from {args.data_filepath} with {len(df)} rows")

    # Initialize the Trainer and start training
    trainer = Trainer(configs, wandb_entity, wandb_project, outputs_dir)
    
    # Set the dataset
    trainer.set_dataset(
        data_df=df,
        image_base_path=args.image_basepath, 
        smiles_col=args.smiles_col,
        activity_col=args.activity_col,
        train_ratio=configs.dataset_configs.train_ratio
    )
    
    # Setup the training environment
    trainer.setup_training()
    
    # Start the training loop
    trainer.train()


if __name__ == "__main__":
    main()