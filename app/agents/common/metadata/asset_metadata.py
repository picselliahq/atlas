import os

import picsellia
from PIL import Image

from services.data_extraction.image_embeddings import CLIPImageEmbedder
from services.embedders import clip_image_embedder


class LocalImageMetadata:
    def __init__(self, file_path: str, image_id: int):
        self.image_id = image_id
        self.target_path = file_path
        self.filename = os.path.basename(file_path)
        self.image = Image.open(file_path)
        self.width = self.image.width
        self.height = self.image.height
        self.clip_embeddings = clip_image_embedder.compute_embeddings(self.image)

    def dict(self):
        """Return a dictionary representation of the metadata."""
        return {
            "filename": self.filename,
            "target_path": self.target_path,
            "width": self.width,
            "height": self.height,
            "clip_embeddings": self.clip_embeddings,
        }


class AssetMetadata:
    def __init__(self, asset: picsellia.Asset):
        self.asset = asset
        self.id = str(asset.id)
        self.filename = asset.filename
        self.target_path = None
        self.width = asset.width
        self.height = asset.height
        self.url = asset.url
        self.tags = self._get_tags()
        self.clip_embeddings = asset._embeddings

    def _get_tags(self):
        """Fetch the tags associated with the asset."""
        try:
            return [e.name for e in self.asset.data_tags]
        except Exception:
            return []

    def get_metadata(self):
        """Returns general metadata about the asset."""
        return {
            "asset_id": self.id,
            "filename": self.filename,
            "target_path": self.target_path,
            "width": self.width,
            "height": self.height,
            "asset_url": self.url,
            "tags": self.tags,
            "clip_embeddings": self.clip_embeddings,
        }
