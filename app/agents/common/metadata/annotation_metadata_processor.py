import logging
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
from picsellia import DatasetVersion, Job
from picsellia.exceptions import (
    BadRequestError,
    InternalServerError,
    WaitingAttemptsTimeout,
)
from tqdm import tqdm

from agents.annotation_agent.common.annotation_clip_embedder import (
    AnnotationCLIPEmbedder,
)
from agents.annotation_agent.common.embeddings import load_all_shapes_and_vectors
from agents.common.metadata.asset_metadata import AssetMetadata
from agents.common.metadata.error_content import get_error_content
from agents.common.models.contents import ReportError
from agents.common.picsellia.list_assets import load_assets
from services.context import ContextService
from services.report_enums import SectionName

logger = logging.getLogger(__name__)


class AnnotationMetadataProcessor:
    def __init__(
        self,
        context_service: ContextService,
        dataset_version: DatasetVersion,
        image_metadata: pd.DataFrame,
    ):
        self.dataset_version = dataset_version
        self.context_service = context_service
        self.image_metadata = image_metadata
        self.embedder = AnnotationCLIPEmbedder(self.dataset_version.type)
        self.embeddings_map: dict[str, list] = {}

    def process(self):
        """Process annotation metadata for all assets in the dataset_version."""
        try:
            try:
                self._get_shape_embeddings_count()
                self.embeddings_map = load_all_shapes_and_vectors(
                    self.dataset_version, 10000
                )
            except (WaitingAttemptsTimeout, BadRequestError):
                message = (
                    "Your embeddings are not ready yet, "
                    "to compute this part of the analysis "
                    "please head to 'Settings' and 'Image Embeddings' "
                    "or 'Shape Embeddings' and check their status"
                )
                content = ReportError(
                    message=message, section=SectionName.ANNOTATION_QUALITY
                )
                self.context_service.sync_content(content)
            assets_to_process = []
            for asset in load_assets(self.dataset_version):
                asset_metadata = AssetMetadata(asset)
                row = self.image_metadata.loc[
                    self.image_metadata["asset_id"] == str(asset.id)
                ]
                target_path = row.iloc[0]["target_path"]
                asset_metadata.target_path = target_path
                assets_to_process.append(asset_metadata)
            data = []
            with ThreadPoolExecutor() as executor:
                results = executor.map(
                    self._process_annotation_metadata, assets_to_process
                )
                for result in tqdm(
                    results, total=len(assets_to_process), desc="Processing assets"
                ):
                    data.extend(result)
            df = pd.DataFrame(data)
        except Exception:
            logger.exception("Error during Annotation dataframe building")
            content = get_error_content(
                context_service=self.context_service,
                dataset_version=self.dataset_version,
            )
            self.context_service.sync_content(content)
            df = pd.DataFrame()
        return df

    def compute_image_embeddings(self):
        resp = self.dataset_version.get_shapes_embeddings_status()
        if resp["success_count"] > int(resp["data_count"] / 2):
            return
        elif len(resp["in_progress_job_runs"]):
            job_payload = resp["in_progress_job_runs"][0]
            job = Job(
                self.dataset_version.connexion,
                {"id": job_payload["job_id"], "status": job_payload["status"]},
                version=2,
            )
            job.wait_for_done(blocking_time_increment=5.0)
        else:
            self.dataset_version.compute_shapes_embeddings()

    def _get_shape_embeddings_count(self):
        """Retrieve count of embeddings, or activate visual search if needed."""
        try:
            self.compute_image_embeddings()
        except BadRequestError:
            logger.exception(
                "Error while activating visual search. "
                "Please check if the dataset version is already activated."
            )
        except InternalServerError:
            logger.exception(
                "Error while activating visual search. "
                "Please check if the dataset version is already activated."
            )
        return self.dataset_version.count_embeddings()

    def _process_annotation_metadata(self, asset_metadata: AssetMetadata):
        """Process annotation metadata for each asset."""
        annotation_id, shapes_metadata = (
            self.embedder.compute_clip_embeddings_from_annotations(
                asset_metadata.asset, self.embeddings_map
            )
        )

        data: list[dict] = []
        if not annotation_id:
            print(f"⚠️ No annotation found for asset {asset_metadata.id}")
            return data
        for _rect_id, shape_metadata in shapes_metadata.items():
            if "x" not in shape_metadata or "y" not in shape_metadata:
                entry = {
                    "filename": asset_metadata.filename,
                    "asset_url": asset_metadata.url,
                    "asset_id": asset_metadata.id,
                    "annotation_id": annotation_id,
                    "shape_id": shape_metadata["id"],
                    "shape_embeddings": shape_metadata["embeddings"],
                    "label": shape_metadata["label"],
                    "label_id": shape_metadata["label_id"],
                    "image_width": asset_metadata.width,
                    "image_height": asset_metadata.height,
                }
            else:
                x_norm, y_norm, w_norm, h_norm = self._normalize_coordinates(
                    shape_metadata, asset_metadata
                )
                entry = {
                    "filename": asset_metadata.filename,
                    "asset_url": asset_metadata.url,
                    "asset_id": asset_metadata.id,
                    "annotation_id": annotation_id,
                    "shape_id": shape_metadata["id"],
                    "shape_embeddings": shape_metadata["embeddings"],
                    "label": shape_metadata["label"],
                    "label_id": shape_metadata["label_id"],
                    "x": shape_metadata["x"],
                    "y": shape_metadata["y"],
                    "w": shape_metadata["w"],
                    "h": shape_metadata["h"],
                    "x_norm": x_norm,
                    "y_norm": y_norm,
                    "w_norm": w_norm,
                    "h_norm": h_norm,
                    "image_width": asset_metadata.width,
                    "image_height": asset_metadata.height,
                }
            data.append(entry)
        return data

    def _normalize_coordinates(self, embedding_data, asset_metadata: AssetMetadata):
        """Normalize coordinates based on image dimensions."""
        x_norm = embedding_data["x"] / asset_metadata.width
        y_norm = embedding_data["y"] / asset_metadata.height
        w_norm = embedding_data["w"] / asset_metadata.width
        h_norm = embedding_data["h"] / asset_metadata.height
        return x_norm, y_norm, w_norm, h_norm
