import logging

import pandas as pd

from atlas_vision.services.context import ContextService
from atlas_vision.services.local_context import LocalContextService

logger = logging.getLogger(__name__)


class PContext:
    def __init__(
        self,
        context_service: ContextService,
        image_metadata: pd.DataFrame,
        annotation_metadata: pd.DataFrame,
    ):
        """Initialize the PContext for annotations."""
        self.context_service = context_service
        self.dataset_id = context_service.dataset_id

        self.metadata = None
        self.final_analysis = None
        self.image_metadata = image_metadata
        self.df = annotation_metadata
        self.local = isinstance(context_service, LocalContextService)
        if not self.local:
            self.client = context_service.client
            self.dataset = self.client.get_dataset_version_by_id(self.dataset_id)

    def get(self, cols: list | None = None) -> pd.DataFrame:
        standard_columns = ["asset_id", "annotation_id"]
        if cols:
            existing_cols = [col for col in cols if col in self.df.columns]
            return self.df[standard_columns + existing_cols]
        return self.df

    def sync(self) -> None:
        """Sync context data to the remote dataset or save it locally."""
        if self.local:
            pass
        else:
            self.context_service.sync_metadata(self.df)
            self.context_service.upload_metadata_to_s3(agent_type="annotation_agent")
            logger.info(f"Context synced to remote dataset: {self.dataset_id}")
