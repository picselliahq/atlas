import json
import os

import pandas as pd

from agents.common.metadata.annotation_metadata import LocalAnnotationMetadata
from agents.common.metadata.asset_metadata import LocalImageMetadata


def compute_analysis(dataset_folder_path: str, coco_file_path: str | None = None):
    """Compute analysis on a dataset with optional COCO annotations.

    Args:
        dataset_folder_path: Path to the dataset directory
        coco_file_path: Optional path to COCO annotations file
    """
    if coco_file_path:
        annotation_data = []
        with open(coco_file_path) as coco_file:
            coco_data = json.load(coco_file)
        images = coco_data["images"]
        annotations = coco_data["annotations"]
        categories = coco_data["categories"]
        labelmap = [category["name"] for category in categories]
        image_map = {}
        image_data = []
        for image in images:
            image_metadata = LocalImageMetadata(
                file_path=os.path.join(dataset_folder_path, image["file_name"]),
                image_id=image["id"],
            )
            image_map[image["id"]] = {
                "image_metadata": image_metadata,
                "annotations": [],
            }
            image_data.append(image_metadata.dict())
        image_df = pd.DataFrame(image_data)
        for annotation in annotations:
            coco_item = {
                "shape_id": annotation["id"],
                "x": annotation["bbox"][0],
                "y": annotation["bbox"][1],
                "w": annotation["bbox"][2],
                "h": annotation["bbox"][3],
                "label_id": annotation["category_id"],
                "label": labelmap[annotation["category_id"]],
                "image_width": images[annotation["image_id"]]["width"],
                "image_height": images[annotation["image_id"]]["height"],
                "filename": images[annotation["image_id"]]["file_name"],
            }
            image_map[annotation["image_id"]]["annotations"].append(coco_item)
            image = image_map[annotation["image_id"]]["image_metadata"].image
            annotation_metadata = LocalAnnotationMetadata(
                coco_item=coco_item,
                image=image,
            )
            annotation_data.append(annotation_metadata.dict())
        annotation_df = pd.DataFrame(annotation_data)
        image_df.to_csv("image_df.csv", index=False)
        annotation_df.to_csv("annotation_df.csv", index=False)
    else:
        filename_list = os.listdir(dataset_folder_path)
        image_paths = [
            os.path.join(dataset_folder_path, filename) for filename in filename_list
        ]
        image_data = []
        for i, image_path in enumerate(image_paths):
            image_metadata = LocalImageMetadata(
                file_path=image_path,
                image_id=i,
            )
            image_data.append(image_metadata)
        image_df = pd.DataFrame(image_data)
        image_df.to_csv("image_df.csv", index=False)
