import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPModelHandler:
    def __init__(self) -> None:
        """Initialize the CLIP model and processor."""
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.mps.is_available()
            else "cpu"
        )
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
            self.device
        )
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def compute_image_embeddings(self, images: list[Image]) -> list[list[float]]:
        """Computes embeddings for a batch of images."""
        batch_size = 64
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                batch_feats = (
                    self.model.get_image_features(**inputs).cpu().numpy().tolist()
                )
            features.extend(batch_feats)
        return features

    def compute_text_embeddings(self, labels: list[str]) -> dict[str, list[float]]:
        """Computes embeddings for a list of text labels."""
        label_embeddings = {}
        for label in labels:
            prompt = f"a {label}"
            inputs = self.processor(text=prompt, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                text_features = (
                    self.model.get_text_features(**inputs).cpu().numpy().tolist()
                )
            label_embeddings[label] = text_features[0]
        return label_embeddings
