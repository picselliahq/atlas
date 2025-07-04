import json
import os
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from atlas_vision.agents.annotation_agent.main import AnnotationAnalysisTool
from atlas_vision.agents.common.metadata.annotation_metadata import (
    LocalAnnotationMetadata,
)
from atlas_vision.agents.common.metadata.asset_metadata import LocalImageMetadata
from atlas_vision.agents.image_agent.main import ImageAnalyticsTool
from atlas_vision.services.local_context import get_local_context_service
from atlas_vision.services.report_enums import SectionName, SubSectionName


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Format DataFrame by converting numpy arrays to lists."""
    for column in df.columns:
        df[column] = df[column].apply(
            lambda x: json.dumps(x[0].tolist())
            if isinstance(x, np.ndarray)
            else json.dumps(x)
        )
    return df


def compute_metadata(dataset_folder_path: str, coco_file_path: str | None = None):
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
        labels = {label["id"]: label["name"] for label in categories}
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
                coco_item=coco_item, image=image, image_id=annotation["image_id"]
            )
            annotation_data.append(annotation_metadata.dict())
        annotation_df = pd.DataFrame(annotation_data)
        image_df = format_dataframe(image_df)
        image_df.to_csv("image_df.csv", index=False)
        annotation_df = format_dataframe(annotation_df)
        annotation_df.to_csv("annotation_df.csv", index=False)
        return image_df, annotation_df, labels
    else:
        filename_list = os.listdir(dataset_folder_path)
        image_paths = [
            os.path.join(dataset_folder_path, filename) for filename in filename_list
        ]
        image_data = []
        for i, image_path in enumerate(image_paths):
            if not os.path.isfile(image_path):
                continue
            image_metadata = LocalImageMetadata(
                file_path=image_path,
                image_id=i,
            )
            image_data.append(image_metadata.dict())
        image_df = pd.DataFrame(image_data)
        for column in image_df.columns:
            image_df[column] = image_df[column].apply(
                lambda x: json.dumps(x[0].tolist())
                if isinstance(x, np.ndarray)
                else json.dumps(x)
            )
        image_df.to_csv("image_df.csv", index=False)
        return image_df, None, None


def compute_analysis(
    dataset_folder_path: str,
    coco_file_path: str | None = None,
    image_df_path: str | None = None,
    annotations_df_path: str | None = None,
):
    """Compute analysis on a dataset with optional COCO annotations.

    Args:
        dataset_folder_path: Path to the dataset directory
        coco_file_path: Optional path to COCO annotations file
        image_df_path: Optional path to the computed image metadata DataFrame
        annotations_df_path: Optional path to the computed annotation metadata DataFrame
    """
    dataset_id = os.path.split(dataset_folder_path)[-1]
    context_service = get_local_context_service(dataset_version_id=dataset_id)
    if image_df_path:
        image_metadata = pd.read_csv(image_df_path)
        for column in image_metadata.columns:
            try:
                image_metadata[column] = image_metadata[column].apply(json.loads)
            except (json.JSONDecodeError, TypeError):
                # Skip columns that are not JSON-serializable
                pass
        if annotations_df_path:
            annotation_metadata = pd.read_csv(annotations_df_path)
            for column in annotation_metadata.columns:
                try:
                    annotation_metadata[column] = annotation_metadata[column].apply(
                        json.loads
                    )
                except (json.JSONDecodeError, TypeError):
                    # Skip columns that are not JSON-serializable
                    pass
            with open(coco_file_path) as coco_file:
                coco_data = json.load(coco_file)
            labels = {label["id"]: label["name"] for label in coco_data["categories"]}
    else:
        image_metadata, annotation_metadata, labels = compute_metadata(
            dataset_folder_path, coco_file_path
        )
    context_service.sync_image_metadata(image_metadata)
    context_service.sync_annotation_metadata(annotation_metadata)
    context_service.sync_labels(labels)
    report = {
        "name": "Report",
        "computing": True,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "sections": [
            {
                "name": SectionName.DATASET_OVERVIEW,
                "sub_sections": [
                    {"name": SubSectionName.OVERVIEW.name},
                ],
            },
            {
                "name": SectionName.IMAGE_QUALITY,
                "sub_sections": [
                    {"name": SubSectionName.IMAGE_STATISTICS_ANALYSIS.name},
                    {"name": SubSectionName.IMAGE_QUALITY_ISSUES.name},
                ],
            },
            {
                "name": SectionName.ANNOTATION_QUALITY,
                "sub_sections": [
                    {"name": SubSectionName.CLASS_ANALYSIS.name},
                    {"name": SubSectionName.SINGLE_OBJECT_ANALYSIS.name},
                    {"name": SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.name},
                    {"name": SubSectionName.INTER_CLASS_RELATION_ANALYSIS.name},
                ],
            },
        ],
        "content": [],
    }
    context_service.sync_context(report)
    ImageAnalyticsTool(context_service=context_service).forward(image_metadata)
    AnnotationAnalysisTool(context_service=context_service).forward(
        image_metadata=image_metadata, annotation_metadata=annotation_metadata
    )
    md_report = context_service._build_md_report()
    with open("report.md", "w") as f:
        f.write(md_report)
