import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPImageEmbedder:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize the CLIP image embedder.

        Args:
            model_name (str): The name of the CLIP model to load. Defaults to "openai/clip-vit-base-patch32"
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def compute_embeddings(
        self, image: Image.Image | list[Image.Image], normalize: bool = True
    ) -> np.ndarray:
        """
        Compute CLIP embeddings for one or multiple images.

        Args:
            image (Union[Image.Image, List[Image.Image]]): Single PIL Image or list of PIL Images
            normalize (bool): Whether to normalize the embeddings. Defaults to True

        Returns:
            np.ndarray: Image embeddings with shape (n_images, embedding_dim)
        """
        # Convert single image to list
        if not isinstance(image, list):
            image = [image]

        # Process images
        inputs = self.processor(images=image, return_tensors="pt", padding=True)

        # Move inputs to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Compute embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Convert to numpy array
        embeddings = image_features.cpu().numpy()

        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings
