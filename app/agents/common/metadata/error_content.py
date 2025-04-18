from picsellia import DatasetVersion

from agents.common.models.contents import ReportError
from services.context import ContextService


def get_error_content(
    context_service: ContextService, dataset_version: DatasetVersion
) -> ReportError:
    message = (
        "There has been an error during the parsing and analysis of your images and labels. \n"
        "Please check that your embeddings are activated and has finished computing with the following links \n"
        "Image embeddings: Go to the Settings tab and then head to the 'Images Embeddings' section  \n"
        "Shape embeddings: Go to the Settings tab and then head to the 'Shapes Embeddings' section  \n"
        "Or contact support: support@picsellia.com"
    )
    return ReportError(message=message)
