import logging
from typing import Any

from picsellia import Asset
from picsellia.types.enums import InferenceType

from .asset_annotation_processor import AssetAnnotationProcessor

# Configure the logger

logger = logging.getLogger(__name__)


class AnnotationCLIPEmbedder:
    def __init__(self, dataset_type: InferenceType) -> None:
        """Initialize the embedder with CLIP model handler and utility handlers."""
        self.annotation_processor = AssetAnnotationProcessor(dataset_type)

    def compute_clip_embeddings_from_annotations(
        self, asset: Asset, embeddings_map: dict
    ) -> tuple[str, dict[str, dict[str, Any]]]:
        """
        Compute CLIP embeddings for each rectangle (or bounding box derived from polygon) in
        the image annotations and return enriched metadata.
        """

        # Step 1: Get the image for the asset
        # image = self.image_handler._get_asset_image(asset)

        # Step 2: Get the first annotation for the asset
        annotation_id, rect_ids, extra_metadata = (
            self.annotation_processor.process_asset(asset=asset)
        )
        if not annotation_id:
            logger.warning(f"⚠️ No annotation found for asset {asset.id}")
            return "", {}
        result = {}
        for rect_id in rect_ids:
            result[rect_id] = {**extra_metadata[rect_id]}
            if rect_id in embeddings_map:
                result[rect_id]["embeddings"] = embeddings_map[rect_id]
            else:
                result[rect_id]["embeddings"] = None
        return annotation_id, result
