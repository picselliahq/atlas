import picsellia
from smolagents import Tool

from services.context import ContextService


class BaseMCPTool(Tool):
    def __init__(self, context: ContextService):
        super().__init__()
        self.context_service: ContextService = context
        self.dataset_version: picsellia.DatasetVersion = (
            self.context_service.client.get_dataset_version_by_id(
                id=self.context_service.dataset_id
            )
        )
