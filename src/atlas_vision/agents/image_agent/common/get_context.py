import logging
from typing import Any

import pandas as pd

from atlas_vision.agents.image_agent.models.analysis_results import (
    ExtendedAnalysisResult,
)
from atlas_vision.agents.image_agent.models.images import ImageGroup, QualityImage
from atlas_vision.agents.image_agent.models.semantic import ExtendedSemanticGroup
from atlas_vision.services.context import ContextService
from atlas_vision.services.local_context import LocalContextService

logger = logging.getLogger(__name__)


class PContext:
    def __init__(
        self,
        context_service: ContextService | LocalContextService,
        image_metadata: pd.DataFrame,
    ):
        self.context_service = context_service
        self.dataset_id = context_service.dataset_id
        self.metadata = None
        self.outlier_groups: list[ImageGroup] = []
        self.contrast_analysis: list[ExtendedAnalysisResult] = []
        self.luminance_analysis: list[ExtendedAnalysisResult] = []
        self.semantic_analysis: list[ExtendedSemanticGroup] = []
        self.quality_analysis: list[QualityImage] = []
        self.data_card: dict[str, Any] = {}
        self.final_analysis: dict[str, Any] | None = None
        self.df = image_metadata
        self.local = isinstance(context_service, LocalContextService)
        if not self.local:
            self.client = context_service.client
            self.dataset = self.client.get_dataset_version_by_id(self.dataset_id)

    def get(self, cols: list | None = None) -> pd.DataFrame:
        standard_columns = ["asset_id", "filename", "asset_url"]
        if cols:
            existing_cols = [col for col in cols if col in self.df.columns]
            all_cols = list(dict.fromkeys(standard_columns + existing_cols))
            return self.df[all_cols]
        return self.df

    def sync(self) -> None:
        if self.local:
            pass
        else:
            self.context_service.sync_metadata(self.df)
            self.context_service.upload_metadata_to_s3(agent_type="image_agent")
            logger.info(f"Context synced to remote dataset: {self.dataset_id}")
