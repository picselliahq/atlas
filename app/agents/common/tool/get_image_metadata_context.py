import pandas as pd
import picsellia
from smolagents import Tool

from services.context import ContextService


class GetImageMetadataTool(Tool):
    name = "get_image_metadata_dataframe"
    description = """
    Retrieve all the information we have about the objects annotated in the Dataset, it returns a Dataframe with these exact columns:
    ["filename","width","height","tags","caption","is_blurry","is_corrupted","blur_score","file_size_bytes","color","luminance","contrast"]

    usage_instructions:
    - "Call this tool when you to answer a question about images in a dataset or compute new charts and analysis based on the content of the dataframe"
    - "Ensure both 'mode' is provided to avoid errors."
    - "The tool will return a dataframe that you can then load and use to answer questions or compute new charts and analysis."
    """
    inputs = {
        "mode": {
            "type": "string",
            "description": "Set mode to 'full' if you want to retrieve all the metadata.",
            "nullable": "True",
        },
    }

    output_type = "object"

    def __init__(self, context: ContextService):
        super().__init__()
        self.context_service: ContextService = context
        self.dataset_version: picsellia.DatasetVersion = (
            self.context_service.client.get_dataset_version_by_id(
                id=self.context_service.dataset_id
            )
        )

    def forward(self, mode: str = "full") -> pd.DataFrame:
        metadata = self.context_service.get_metadata(agent_type="image_agent")
        return self.context_service._filter_image_dataframe_columns(metadata)
