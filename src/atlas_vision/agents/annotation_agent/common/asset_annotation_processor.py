import logging
from typing import Any

import picsellia
from picsellia.types.enums import InferenceType
from PIL import Image

logger = logging.getLogger(__name__)


class AssetAnnotationProcessor:
    def __init__(self, dataset_type: InferenceType) -> None:
        self.dataset_type = dataset_type

    def _get_first_annotation(
        self, asset: picsellia.Asset
    ) -> picsellia.Annotation | None:
        """Fetch the first annotation for the asset."""
        try:
            return asset.list_annotations()[0]
        except IndexError:
            logger.warning(f"⚠️ No annotation found for asset: {asset.id}")
            return None

    def _process_annotation(
        self, annotation: picsellia.Annotation
    ) -> list[dict[str, Any]]:
        """Uniformly process annotations, whether they are rectangles or polygons."""
        processed_annotations = []
        if self.dataset_type == InferenceType.OBJECT_DETECTION:
            for rect in annotation.list_rectangles():
                processed_annotations.append(
                    {
                        "x": rect.x,
                        "y": rect.y,
                        "w": rect.w,
                        "h": rect.h,
                        "label": rect.label.name,
                        "label_id": str(rect.label.id),
                        "id": str(rect.id),
                    }
                )
        elif self.dataset_type == InferenceType.SEGMENTATION:
            for polygon in annotation.list_polygons():
                x, y, w, h = self._polygon_to_bbox(polygon)
                processed_annotations.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "label": polygon.label.name,
                        "label_id": str(polygon.label.id),
                        "id": str(polygon.id),
                    }
                )
        elif self.dataset_type == InferenceType.CLASSIFICATION:
            for classification in annotation.list_classifications():
                processed_annotations.append(
                    {
                        "label": classification.label.name,
                        "label_id": str(classification.label.id),
                        "id": str(classification.id),
                    }
                )

        return processed_annotations

    def _polygon_to_bbox(self, polygon: picsellia.Polygon) -> tuple[int, int, int, int]:
        """Converts polygon coordinates to a bounding box."""
        min_x = min(polygon.coords, key=lambda p: p[0])[0]
        max_x = max(polygon.coords, key=lambda p: p[0])[0]
        min_y = min(polygon.coords, key=lambda p: p[1])[1]
        max_y = max(polygon.coords, key=lambda p: p[1])[1]
        return min_x, min_y, max_x - min_x, max_y - min_y

    def _prepare_crops(
        self, annotations: list[dict[str, Any]]
    ) -> tuple[list[str], dict[str, dict[str, Any]]]:
        """Prepare crops from bounding boxes."""
        rect_ids = []
        extra_metadata = {}

        for ann in annotations:
            if "x" in ann and "y" in ann:
                w, h = ann["w"], ann["h"]
                if w * h <= 100:  # Ignore very small rectangles
                    continue

                rect_id = f"{ann['id']}"
                rect_ids.append(rect_id)

                extra_metadata[rect_id] = {
                    "id": ann["id"],
                    "x": ann["x"],
                    "y": ann["y"],
                    "w": ann["w"],
                    "h": ann["h"],
                    "label": ann["label"],
                    "label_id": ann["label_id"],
                }
            else:
                classification_id = ann["id"]
                rect_ids.append(classification_id)

                extra_metadata[classification_id] = {
                    "id": classification_id,
                    "label": ann["label"],
                    "label_id": ann["label_id"],
                    # "image_width": image.width,
                    # "image_height": image.height,
                }

        return rect_ids, extra_metadata

    def _pad_to_224(self, image: Image.Image) -> Image.Image:
        """Pad image to at least 224x224."""
        width, height = image.size
        new_width = max(224, width)
        new_height = max(224, height)
        padded = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded.paste(image, ((new_width - width) // 2, (new_height - height) // 2))
        return padded

    def process_asset(
        self, asset: picsellia.Asset
    ) -> tuple[str, list[str], dict[str, dict[str, Any]]]:
        """Main method to process an asset and its annotations."""
        annotation = self._get_first_annotation(asset)
        if not annotation:
            return "", [], {}

        annotations = self._process_annotation(annotation)

        rect_ids, extra_metadata = self._prepare_crops(annotations)

        return str(annotation.id), rect_ids, extra_metadata
