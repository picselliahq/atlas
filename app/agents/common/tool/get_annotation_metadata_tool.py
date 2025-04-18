import pandas as pd
import picsellia
from smolagents import Tool

from services.context import ContextService


class GetAnnotationMetadataTool(Tool):
    name = "get_annotation_metadata_dataframe"
    description = """
    Retrieve all the information we have about the objects annotated in the Dataset, it returns a Dataframe with these exact columns:
    ["filename","image_width","image_height","label","x","y","w","h","x_norm","y_norm"]
    if you want to analyse all the annotated objects from an Image, you should use df.groupby("filename")
    usage_instructions:
    - "Call this tool when you to answer a question about shapes and labels in a dataset or compute new charts and analysis based on the content of the dataframe"
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
        metadata = self.context_service.get_metadata(agent_type="annotation_agent")
        return self.context_service._filter_annotation_dataframe_columns(metadata)
