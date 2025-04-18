import gc

import picsellia
import requests
import torch
from PIL import Image

from agents.common.ai_models.captioning_model import CaptioningModel


class AssetCaptioning:
    def __init__(self) -> None:
        """Initialize the captioning process with the CaptioningModel."""
        self.captioning_model = CaptioningModel()

    def get_image_from_asset(self, asset: picsellia.Asset) -> Image.Image:
        """Fetch the image from the asset URL."""
        try:
            img_url = asset.url
            return Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        except Exception as e:
            raise Exception(f"Error fetching image from asset: {e}") from e

    def caption(self, asset: picsellia.Asset) -> str:
        """
        Generate captions for an asset. In the future, we may extend this to support captioning based on labels.
        """
        try:
            image = self.get_image_from_asset(asset)

            caption = self.captioning_model.generate_caption(image)

            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

            return caption
        except Exception as e:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            raise e
