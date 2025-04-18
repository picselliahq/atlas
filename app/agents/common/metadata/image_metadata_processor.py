import logging
import os
import shutil
import uuid
from concurrent.futures.thread import ThreadPoolExecutor

import pandas as pd
from picsellia import DatasetVersion
from picsellia.exceptions import (
    BadRequestError,
    InternalServerError,
    WaitingAttemptsTimeout,
)
from tqdm import tqdm

from agents.common.metadata.asset_metadata import AssetMetadata
from agents.common.metadata.error_content import get_error_content
from agents.common.models.contents import ReportError
from agents.common.picsellia.list_assets import load_assets
from agents.image_agent.common.embeddings import load_all_assets_and_vectors
from agents.image_agent.common.image_blurriness_compute import (
    assess_bluriness_and_corruption,
)
from services.context import ContextService
from services.report_enums import SectionName

logger = logging.getLogger(__name__)


class ImageMetadataProcessor:
    def __init__(
        self, context_service: ContextService, dataset_version: DatasetVersion
    ):
        self.context_service = context_service
        self.dataset_version = dataset_version

    def process(self):
        """Processes image metadata for all assets in the dataset version."""
        target_path = f"{uuid.uuid4()}"
        self.dataset_version.download(target_path)
        try:
            assets = load_assets(self.dataset_version)
            try:
                count = self._get_dataset_version_count()
                assets = load_all_assets_and_vectors(
                    assets=assets, dataset_version=self.dataset_version, count=count
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

            # Prepare assets for processing
            assets_to_process = []
            for asset in assets:
                asset_obj = AssetMetadata(asset)
                asset_obj.target_path = os.path.join(target_path, asset.filename)
                assets_to_process.append(asset_obj)

            with ThreadPoolExecutor() as executor:
                data = list(
                    tqdm(
                        executor.map(self._process_asset_metadata, assets_to_process),
                        total=len(assets_to_process),
                        desc="Processing assets",
                    )
                )

        except Exception:
            logger.exception("Error during parallel processing")
            content = get_error_content(
                context_service=self.context_service,
                dataset_version=self.dataset_version,
            )
            self.context_service.sync_content(content)
            return pd.DataFrame()

        finally:
            shutil.rmtree(target_path, ignore_errors=True)
        return pd.DataFrame(data)

    def compute_image_embeddings(self):
        try:
            self.dataset_version.activate_visual_search()
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

    def _get_dataset_version_count(self):
        """Retrieve count of embeddings, or activate visual search if needed."""
        try:
            return self.dataset_version.count_embeddings()
        except Exception:
            self.compute_image_embeddings()
            return self.dataset_version.count_embeddings()

    @staticmethod
    def _process_asset_metadata(asset_metadata: AssetMetadata):
        """Process metadata related to the image."""
        try:
            (
                is_blurry,
                is_corrupted,
                blur_score,
                width,
                height,
                file_size_bytes,
                avg_color,
                luminance_value,
                contrast,
            ) = assess_bluriness_and_corruption(
                filename=asset_metadata.target_path, blur_threshold=90.0
            )

            return {
                **asset_metadata.get_metadata(),
                "caption": "",
                "is_blurry": is_blurry,
                "is_corrupted": is_corrupted,
                "blur_score": blur_score,
                "width": width,
                "height": height,
                "file_size_bytes": file_size_bytes,
                "color": avg_color,
                "luminance": luminance_value,
                "contrast": contrast,
            }
        except Exception as e:
            logger.error(
                f"Error processing asset {asset_metadata.asset.filename}: {str(e)}"
            )
            raise
