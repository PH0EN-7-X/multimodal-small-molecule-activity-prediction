import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from huggingface_mae import MAEModel

from peft import (
    IA3Config,
    LoHaConfig,
    LoKrConfig,
    LoraConfig,
    TaskType,
    get_peft_model,
)

from cellberta.configs import ModelConfigs
from cellberta.models.base_model import BaseModel


PEFT_TYPE_TO_CONFIG_MAPPING = {
    "lora": LoraConfig,
    "loha": LoHaConfig,
    "lokr": LoKrConfig,
    "ia3": IA3Config,
}

def get_peft_config(peft_type):
    return PEFT_TYPE_TO_CONFIG_MAPPING[peft_type]


class CellBERTa(BaseModel):
    
    """
    CellBERTa model using cellular image transformers and ligand language models, 
    projecting them into a shared space using a cosine similarity metric for compatibility.

    Args:
        model_configs (ModelConfigs): Configuration object for the model, containing all necessary hyperparameters.
        image_embedding_size (int): Size of the image embedding (default=768).
        drug_embedding_size (int): Size of the drug embedding (default=384).
    """

    def __init__(
        self,
        model_configs: ModelConfigs,
        image_embedding_size=768,
        drug_embedding_size=384,
    ):
        # Initialize the base model with configurations and embedding sizes
        super(CellBERTa, self).__init__(
            model_configs, image_embedding_size, drug_embedding_size
        )

        # Set ReLU activation before cosine similarity if specified in the model configs
        self.relu_before_cosine = (
            self.model_configs.model_hyperparameters.relu_before_cosine
        )

        # Apply Image PEFT (Parameter Efficient Fine-Tuning) configuration if provided
        if model_configs.image_peft_hyperparameters:
            # Retrieve and apply image PEFT configurations
            self.image_peft_config = get_peft_config(
                model_configs.image_fine_tuning_type
            )(
                **model_configs.image_peft_hyperparameters,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.image_model = get_peft_model(
                self.image_model, self.image_peft_config
            )
            # Print trainable parameters in the image model
            self.image_model.print_trainable_parameters()

        # Apply Drug PEFT configuration if specified in model configs
        if model_configs.drug_peft_hyperparameters:
            # Retrieve and apply drug PEFT configurations
            self.drug_peft_config = get_peft_config(
                model_configs.drug_fine_tuning_type
            )(
                **model_configs.drug_peft_hyperparameters,
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.drug_model = get_peft_model(self.drug_model, self.drug_peft_config)
            # Print trainable parameters in the drug model
            self.drug_model.print_trainable_parameters()

        # Define linear projection layers for image and drug embeddings
        self.image_projection = nn.Linear(
            self.image_embedding_size,
            model_configs.model_hyperparameters.projected_size,
        )
        self.drug_projection = nn.Linear(
            self.drug_embedding_size, model_configs.model_hyperparameters.projected_size
        )

        # Apply dropout with a rate defined in model configs
        self.dropout = nn.Dropout(model_configs.model_hyperparameters.projected_dropout)

        # Define loss function and type
        self.loss_fn_type = model_configs.loss_function
        self.loss_fn = nn.MSELoss()

    @staticmethod
    def cosine_similarity_to_activity(cosine_similarity, activity_upper_bound, activity_lower_bound):
        """
        Converts cosine similarity scores to activity values by scaling within defined bounds.

        Args:
            cosine_similarity (Tensor): Cosine similarity score(s) between image and drug embeddings.
            activity_upper_bound (float): Maximum activity value.
            activity_lower_bound (float): Minimum activity value.

        Returns:
            Tensor: Scaled activity values.
        """
        activity_range = activity_upper_bound - activity_lower_bound
        return (cosine_similarity + 1) / 2 * activity_range + activity_lower_bound

    def forward(self, batch_input, **kwargs):
        """
        Forward pass of the CellBERTa model.

        Args:
            batch_input (dict): Input batch containing image and drug input data.
            **kwargs: Additional keyword arguments for flexibility.

        Returns:
            dict: Output dictionary containing cosine similarity, embeddings, and optional loss.
        """
        forward_output = {}

        # Extract image embeddings by processing the input image
        if "image_input" in batch_input:  # When directly feeding image embeddings
            image_embedding = batch_input["image_input"]
        else:  # When processing raw image data
            image_embedding = self.image_model.predict(batch_input["image_tensor"])
            
        # Apply dropout to the image embedding and project it to shared space
        image_embedding = self.dropout(image_embedding)
        image_embedding = self.image_projection(image_embedding)

        # Extract drug embeddings using ChemBERTa
        drug_embedding = self.drug_model(
            input_ids=batch_input["drug_input_ids"],
            attention_mask=batch_input["drug_attention_mask"],
        )["pooler_output"]
        
        # Apply dropout to the drug embedding and project it to shared space
        drug_embedding = self.dropout(drug_embedding)
        drug_embedding = self.drug_projection(drug_embedding)

        # Optionally apply ReLU activation to both embeddings before cosine similarity
        if self.relu_before_cosine:
            image_embedding = F.relu(image_embedding)
            drug_embedding = F.relu(drug_embedding)

        # Compute cosine similarity between image and drug embeddings
        cosine_similarity = F.cosine_similarity(image_embedding, drug_embedding)

        # Calculate loss if labels are provided in the batch input
        if "labels" in batch_input:
            if batch_input["labels"] is not None:
                forward_output["loss"] = self.loss_fn(
                    cosine_similarity, batch_input["labels"]
                )

        # Store outputs in the dictionary
        forward_output["cosine_similarity"] = cosine_similarity
        forward_output["image_embedding"] = image_embedding
        forward_output["drug_embedding"] = drug_embedding

        return forward_output