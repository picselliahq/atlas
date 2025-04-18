import logging
from typing import Any

import pandas as pd
import picsellia
import requests

from agents.common.metadata.image_metadata_processor import ImageMetadataProcessor
from agents.image_agent.models.analysis_results import ExtendedAnalysisResult
from agents.image_agent.models.images import ImageGroup, QualityImage
from agents.image_agent.models.semantic import ExtendedSemanticGroup
from services.context import ContextService

logger = logging.getLogger(__name__)


class PContext:
    def __init__(
        self,
        context_service: ContextService,
        ctx_path: str | None = None,
        local: bool = False,
        force_recompute: bool = True,
    ):
        self.ctx_path = ctx_path
        self.context_service = context_service
        self.dataset_id = context_service.dataset_id
        self.local = local
        self.metadata = None
        self.outlier_groups: list[ImageGroup] = []
        self.contrast_analysis: list[ExtendedAnalysisResult] = []
        self.luminance_analysis: list[ExtendedAnalysisResult] = []
        self.semantic_analysis: list[ExtendedSemanticGroup] = []
        self.quality_analysis: list[QualityImage] = []
        self.data_card: dict[str, Any] = {}
        self.final_analysis: dict[str, Any] | None = None
        self.client = context_service.client

        if local:
            self._load_local_context()
        else:
            self._load_remote_context(force_recompute)

    def _load_local_context(self):
        if self.ctx_path and self.ctx_path.endswith(".csv"):
            self.df = pd.read_csv(self.ctx_path)
            logger.info(f"Loaded context from {self.ctx_path}.")
        else:
            logger.error("Invalid context_path or file format, compute context first.")
            raise ValueError(
                "Invalid context_path or file format, compute context first."
            )

    def _load_remote_context(self, force_recompute):
        self.dataset = self.client.get_dataset_version_by_id(self.dataset_id)
        if force_recompute:
            self.generate_context(self.dataset)
        else:
            try:
                self.df = self.context_service.get_metadata(agent_type="image_agent")
            except requests.exceptions.HTTPError as e:
                logger.error(e)
                self.generate_context(self.dataset)
                logger.info("Generated context from remote dataset.")

            try:
                # self.final_analysis = self.context_service.get_context_dict()
                self.final_analysis = None
                logger.info("Loaded final analysis from remote dataset.")
            except requests.exceptions.HTTPError:
                logger.info("Final analysis not found remotely.")
                self.final_analysis = None

    def generate_context(self, dataset: picsellia.DatasetVersion = None) -> None:
        self.df = ImageMetadataProcessor(
            context_service=self.context_service, dataset_version=dataset
        ).process()
        self.context_service.sync_metadata(self.df)
        self.context_service.upload_metadata_to_s3(agent_type="image_agent")

    def get(self, cols: list | None = None) -> pd.DataFrame:
        standard_columns = ["asset_id", "filename", "asset_url"]
        if cols:
            existing_cols = [col for col in cols if col in self.df.columns]
            all_cols = list(dict.fromkeys(standard_columns + existing_cols))
            return self.df[all_cols]
        return self.df

    def sync(self) -> None:
        if self.local:
            self.df.to_csv(self.ctx_path, index=False)
            logger.info(f"Context synced to local file: {self.ctx_path}")
        else:
            self.context_service.sync_metadata(self.df)
            self.context_service.upload_metadata_to_s3(agent_type="image_agent")
            logger.info(f"Context synced to remote dataset: {self.dataset_id}")
