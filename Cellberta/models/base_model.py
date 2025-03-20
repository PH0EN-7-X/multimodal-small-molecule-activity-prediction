import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel
from huggingface_mae import MAEModel
from cellberta.configs import ModelConfigs


class BaseModel(nn.Module):
    """
    BaseModel integrates both cell image vision transformer and ligand language models.
    
    Attributes:
        model_configs (ModelConfigs): The configuration object for the model.
        image_model: The pre-trained vision transformer model for cell images.
        drug_model (AutoModel): The pre-trained ligand language model.
        image_embedding_size (int): The size of the image model embeddings.
        drug_embedding_size (int): The size of the drug model embeddings.
    """

    def __init__(self, model_configs, image_embedding_size, drug_embedding_size):
        """
        Initializes the BaseModel with the given configuration.

        Args:
            model_configs (ModelConfigs): The configuration object for the model.
            image_embedding_size (int): The size of the image embeddings.
            drug_embedding_size (int): The size of the drug embeddings.
        """
        super(BaseModel, self).__init__()
        self.model_configs = model_configs

        # Initialize OpenPhenom vision transformer model
        self.image_model = MAEModel.from_pretrained(
            model_configs.image_model_name_or_path,
            device_map="auto",
        )
        self.image_model.return_channelwise_embeddings = True  # Enable channel-wise embeddings
        
        # Initialize ChemBERTa model for drug embeddings
        self.drug_model = AutoModel.from_pretrained(
            model_configs.drug_model_name_or_path,
            device_map="auto",
        )

        # Freeze all parameters initially
        for name, param in self.image_model.named_parameters():
            param.requires_grad = False

        for name, param in self.drug_model.named_parameters():
            param.requires_grad = False

        # Set pooler layers to be trainable
        self._set_pooler_layer_to_trainable()

        self.image_embedding_size = image_embedding_size
        self.drug_embedding_size = drug_embedding_size

    def _set_pooler_layer_to_trainable(self):
        """
        Manually sets the pooler layer to be trainable for both image and drug models.
        """
        # For the drug model (ChemBERTa), enable training of the pooler layer
        for name, param in self.drug_model.named_parameters():
            if "pooler.dense" in name:
                param.requires_grad = True
                
        # For OpenPhenom, enable last few layers to be fine-tuned if needed
        # This will depend on the specific architecture of the MAE model
        try:
            for name, param in self.image_model.named_parameters():
                if "decoder" in name:  # Typically, decoder layers are more task-specific
                    param.requires_grad = True
        except AttributeError:
            # If the MAE model doesn't have the expected structure, we'll skip this step
            pass

    def print_trainable_params(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                print(name)
                trainable_params += num_params

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )