import logging

import pandas as pd
import picsellia
import requests

from agents.common.metadata.annotation_metadata_processor import (
    AnnotationMetadataProcessor,
)
from services.context import ContextService

logger = logging.getLogger(__name__)


class PContext:
    def __init__(
        self,
        context_service: ContextService,
        image_metadata: pd.DataFrame,
        force_recompute: bool = True,
        ctx_path: str | None = None,
        local: bool = False,
    ):
        """Initialize the PContext for annotations."""
        self.ctx_path = ctx_path
        self.context_service = context_service
        self.dataset_id = context_service.dataset_id
        self.local = local
        self.metadata = None
        self.final_analysis = None
        self.client = context_service.client
        self.image_metadata = image_metadata
        if local:
            self._load_local_context()
        else:
            self._load_remote_context(force_recompute)

    def _load_local_context(self):
        """Load local context from the CSV file."""
        if self.ctx_path and self.ctx_path.endswith(".csv"):
            self.df = pd.read_csv(self.ctx_path)
            logger.info(f"Loaded context from {self.ctx_path}.")
        else:
            logger.error("Invalid context_path or file format, compute context first.")
            raise ValueError(
                "Invalid context_path or file format, compute context first."
            )

    def _load_remote_context(self, force_recompute):
        """Load context remotely from the ContextService."""
        self.dataset = self.client.get_dataset_version_by_id(self.dataset_id)
        if force_recompute:
            self.generate_context(self.dataset)
        else:
            try:
                self.df = self.context_service.get_metadata(
                    agent_type="annotation_agent"
                )
                logger.info("Loaded context metadata from remote.")
            except requests.exceptions.HTTPError as e:
                logger.error(f"Error loading context metadata: {e}")
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
        """Generate context metadata for the annotation dataset."""
        self.df = AnnotationMetadataProcessor(
            context_service=self.context_service,
            dataset_version=dataset,
            image_metadata=self.image_metadata,
        ).process()
        # Use ContextService to sync metadata and upload to remote storage
        self.context_service.sync_metadata(self.df)
        self.context_service.upload_metadata_to_s3(agent_type="annotation_agent")

    def get(self, cols: list | None = None) -> pd.DataFrame:
        standard_columns = ["asset_id", "annotation_id"]
        if cols:
            existing_cols = [col for col in cols if col in self.df.columns]
            return self.df[standard_columns + existing_cols]
        return self.df

    def sync(self) -> None:
        """Sync context data to the remote dataset or save it locally."""
        if self.local:
            self.df.to_csv(self.ctx_path, index=False)
            logger.info(f"Context synced to local file: {self.ctx_path}")
        else:
            self.context_service.sync_metadata(self.df)
            self.context_service.upload_metadata_to_s3(agent_type="annotation_agent")
            logger.info(f"Context synced to remote dataset: {self.dataset_id}")
