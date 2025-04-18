import picsellia
from smolagents import Tool

from services.context import ContextService


class GetDatasetAnalysisReport(Tool):
    name = "get_dataset_analysis_report"
    description = """
    Retrieve all the analysis we have about the dataset you are currently analysing.
    The report contains informations about: Overview, Image Quality (Blur, Contrast, Luminanc, CLIP outliers), Annotation Quality (Object shapes, overlap, co-occurence, etc.)

    usage_instructions:
    - Do not call this tool unless you are looking for a specific information.
    - "Ensure 'mode' is provided to avoid errors."
    - "The tool will return a string markdown formatted."
    """
    inputs = {
        "mode": {
            "type": "string",
            "description": "set to full.",
            "nullable": "True",
        },
    }

    output_type = "string"

    def __init__(self, context: ContextService):
        super().__init__()
        self.context_service: ContextService = context
        self.dataset_version: picsellia.DatasetVersion = (
            self.context_service.client.get_dataset_version_by_id(
                id=self.context_service.dataset_id
            )
        )

    def forward(self, mode: str = "full") -> str:
        metadata = self.context_service._build_md_report()
        return metadata
