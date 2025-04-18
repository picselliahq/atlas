from picsellia import DatasetVersion
from smolagents import Tool

from services.context import ContextService


class PicselliaBaseTool(Tool):
    def __init__(self, context_service: ContextService, **kwargs) -> None:
        super().__init__(**kwargs)
        self.context_service = context_service
        self.client = context_service.client
        self.dataset: DatasetVersion = self.client.get_dataset_version_by_id(
            context_service.dataset_id
        )
