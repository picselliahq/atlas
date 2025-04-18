import logging

from pydantic_ai import Agent

from agents.ai_models import report_model
from agents.common.models.contents import ReportContent
from agents.image_agent.analysis.semantic_analysis.interpreter import (
    SemanticCaptionInterpreter,
)
from agents.image_agent.analysis.semantic_analysis.stats import SemanticCaptionStats
from agents.image_agent.common.get_context import PContext
from agents.image_agent.models.semantic import (
    SemanticGroup,
)
from services.report_enums import SubSectionName

system_prompt = """
TASK: Analyze semantic clusters of image captions to extract key dataset insights and format results according to specified Pydantic models

CONTEXT:
You are assisting a computer vision engineer who has performed DBSCAN clustering on image captions using TF-IDF vectorization. These clusters represent semantically similar captions within the dataset.

OBJECTIVE:
Analyze the provided caption clusters to identify:
1. The overall purpose/goal of the image dataset
2. Potential objects that could be pre-annotated for model training (with NO overlap between semantic groups)
3. Images that appear to be outside the dataset's scope (outliers to be removed)

INPUT FORMAT:
You will receive a list of caption clusters, where each cluster contains semantically similar captions grouped by the DBSCAN algorithm. Each cluster will be provided as a JSON object with a cluster ID and associated captions.

EXPECTED ANALYSIS:
For each semantic cluster, analyze the captions to extract meaningful insights about their content and relationship to the overall dataset.

ADDITIONAL GUIDELINES:

CRITICAL: Ensure NO overlap in named_entities across different semantic groups
Before finalizing output, cross-check all named_entities lists to eliminate duplicates
If an object could belong to multiple groups, make a decisive assignment to only one group based on prevalence and relevance
Prioritize objects that are uniquely characteristic of each semantic group
When resolving overlaps, prefer more specific terms (e.g., "sports car" vs "car")
For out_of_scope_entities, focus on entities that genuinely don't belong in the dataset rather than rare but relevant entities
Ensure every field in the model is populated according to the specifications
Ensure every field in the model is populated according to the specifications
Format your final output as a list of ExtendedSemanticGroup objects in valid JSON
"""

logger = logging.getLogger(__name__)
semantic_analyzer_agent = Agent(
    model=report_model,
    system_prompt=system_prompt,
    result_type=list[SemanticGroup],
    retries=3,
)


def extract_semantic_concepts(pctx: PContext) -> ReportContent:
    df = pctx.get(cols=["caption", "asset_id"])

    logger.info("📊 Running semantic analysis...")
    stats = SemanticCaptionStats(df=df, eps=0.5, min_samples=2, top_words=10).compute()

    logger.info("🧠 Running LLM interpretation for semantic analysis...")
    content = SemanticCaptionInterpreter(
        stats=stats, agent=semantic_analyzer_agent
    ).run(
        section=SubSectionName.CAPTION_CLUSTERING.section.value,
        sub_section=SubSectionName.CAPTION_CLUSTERING.name,
    )
    pctx.context_service.sync_content(content)
    return content
