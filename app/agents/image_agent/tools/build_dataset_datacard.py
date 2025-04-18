import logging

from pydantic_ai import Agent

from agents.ai_models import report_model
from agents.common.models.contents import ReportContent, ReportError
from agents.image_agent.analysis.data_card.interpreter import DataCardInterpreter
from agents.image_agent.analysis.data_card.stats import DataCardStats
from agents.image_agent.common.get_context import PContext
from agents.image_agent.models.datacard import DataCard
from services.report_enums import ReportContentName

datacard_generator = Agent(
    model=report_model,
    result_type=DataCard,
    system_prompt="You are a specialized DataCard Builder agent. You will receive a dictionary containing metadata about a dataset. Your task is to parse the dictionary and produce a DataCard",
    retries=3,
)

logger = logging.getLogger(__name__)


def build_dataset_datacard(pctx: PContext) -> ReportContent | ReportError:
    try:
        stats = DataCardStats(dataset_version=pctx.dataset).compute()
        content = DataCardInterpreter(stats=stats, agent=datacard_generator).run(
            section=ReportContentName.METADATA_OVERVIEW.section,
            sub_section=ReportContentName.METADATA_OVERVIEW.sub_section,
            content_name=ReportContentName.METADATA_OVERVIEW.content,
        )
    except Exception as e:
        logger.warning(f"Failed to compute metadata: {e}")
        content = ReportError(
            section=ReportContentName.METADATA_OVERVIEW.section,
            message="Failed to compute metadata",
        )
    pctx.context_service.sync_content(content)
    return content
