import logging

from agents.common.models.contents import ReportContent, ReportError
from agents.image_agent.analysis.clip.interpreter import (
    DuplicateInterpreter,
    OutlierInterpreter,
)
from agents.image_agent.analysis.clip.stats import ClipStats
from agents.image_agent.common.get_context import PContext
from services.report_enums import ReportContentName

logger = logging.getLogger(__name__)


def detect_clip_outliers(pctx: PContext) -> list[ReportContent | ReportError]:
    df = pctx.get(cols=["asset_id", "clip_embeddings"])
    try:
        stats = ClipStats(
            df=df, top_percent_outlier=3.0, dbscan_eps=0.005, dbscan_min_samples=2
        ).compute()
    except Exception as e:
        logger.warning(f"Failed to compute clip outliers: {e}")
        return [
            ReportError(
                section=ReportContentName.CLIP_OUTLIER_ANALYSIS.section,
                message="Failed to compute clip outliers",
            )
        ]
    try:
        outliers_report = OutlierInterpreter(stats=stats).run(
            section=ReportContentName.CLIP_OUTLIER_ANALYSIS.section,
            sub_section=ReportContentName.CLIP_OUTLIER_ANALYSIS.sub_section,
            content_name=ReportContentName.CLIP_OUTLIER_ANALYSIS.content,
        )
    except Exception as e:
        logger.warning(f"Failed to interpret clip outliers: {e}")
        return [
            ReportError(
                section=ReportContentName.CLIP_OUTLIER_ANALYSIS.section,
                sub_section=ReportContentName.CLIP_OUTLIER_ANALYSIS.sub_section,
                message="Failed to interpret clip outliers",
            )
        ]
    try:
        duplicates_report = DuplicateInterpreter(stats=stats).run(
            section=ReportContentName.CLIP_DUPLICATE_ANALYSIS.section,
            sub_section=ReportContentName.CLIP_DUPLICATE_ANALYSIS.sub_section,
            content_name=ReportContentName.CLIP_DUPLICATE_ANALYSIS.content,
        )
    except Exception as e:
        logger.warning(f"Failed to interpret duplicate analysis: {e}")
        return [
            ReportError(
                section=ReportContentName.CLIP_DUPLICATE_ANALYSIS.section,
                sub_section=ReportContentName.CLIP_DUPLICATE_ANALYSIS.sub_section,
                message="Failed to interpret duplicate analysis",
            )
        ]
    pctx.context_service.sync_content(outliers_report)
    pctx.context_service.sync_content(duplicates_report)
    return [outliers_report, duplicates_report]
