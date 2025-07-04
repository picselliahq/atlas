import os

import cv2
import numpy as np
import picsellia
from PIL import Image

from atlas_vision.services.embedders import clip_image_embedder


class LocalImageMetadata:
    def __init__(self, file_path: str, image_id: int):
        self.image_id = image_id
        self.target_path = file_path
        self.filename = os.path.basename(file_path)
        self.image = Image.open(file_path)
        self.width = self.image.width
        self.height = self.image.height
        self.clip_embeddings = clip_image_embedder.compute_embeddings(self.image)
        try:
            self.luminance, self.contrast, self.avg_color = (
                self.get_luminance_contrast_avg_color(self.image)
            )
            self.is_blurry, self.blur_score = self.get_blur_score(self.image)
            self.is_corrupted = False
        except Exception as e:
            self.luminance = None
            self.contrast = None
            self.avg_color = None
            self.is_blurry = True
            self.blur_score = 0.0
            self.is_corrupted = True
            print(f"Error processing image {self.filename}: {e}")

    def dict(self):
        """Return a dictionary representation of the metadata."""
        return {
            "asset_id": self.filename,
            "asset_url": self.target_path,
            "filename": self.filename,
            "target_path": self.target_path,
            "width": self.width,
            "height": self.height,
            "clip_embeddings": self.clip_embeddings,
            "luminance": self.luminance,
            "contrast": self.contrast,
            "avg_color": self.avg_color,
            "is_blurry": self.is_blurry,
            "blur_score": self.blur_score,
            "is_corrupted": self.is_corrupted,
        }

    @staticmethod
    def get_luminance_contrast_avg_color(image: Image):
        image_array = np.array(image)
        luminance = (
            0.2126 * image_array[:, :, 0]
            + 0.7152 * image_array[:, :, 1]
            + 0.0722 * image_array[:, :, 2]
        )
        luminance_value = luminance.mean(axis=(0, 1))
        contrast = np.std(luminance)
        avg_color = image_array.mean(axis=(0, 1))
        return luminance_value, contrast, avg_color

    @staticmethod
    def get_blur_score(image: Image, blur_threshold: float = 90.0):
        image_cv = np.array(image)
        # Convert to grayscale if the image is colored (3 or 4 channels)
        if len(image_cv.shape) == 3 and image_cv.shape[2] in [3, 4]:
            image_gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
        else:
            image_gray = image_cv
        blur_score = cv2.Laplacian(image_gray, cv2.CV_64F).var()
        is_blurry = blur_score < blur_threshold
        return is_blurry, blur_score


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
