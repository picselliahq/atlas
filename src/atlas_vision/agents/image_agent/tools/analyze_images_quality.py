import logging

import pandas as pd
from pydantic_ai import Agent

from atlas_vision.agents.ai_models import report_model
from atlas_vision.agents.common.interpreter.analysis.multi_chart_stats_interpreter import (
    MultiChartStatsInterpreter,
)
from atlas_vision.agents.common.interpreter.outlier.outlier_interpreter import (
    OutlierInterpreter,
)
from atlas_vision.agents.common.models.contents import ReportContent, ReportError
from atlas_vision.agents.common.utils import sync_reports
from atlas_vision.agents.image_agent.analysis.image_quality.outliers import (
    ImageQualityOutlierStats,
)
from atlas_vision.agents.image_agent.analysis.image_quality.stats import (
    ImageQualityStats,
)
from atlas_vision.agents.image_agent.common.get_context import PContext
from atlas_vision.agents.prompts.generator import PromptGenerator
from atlas_vision.services.report_enums import ReportContentName

analysis_type = """
Analyse both a Luminance histogram chart, a Contrast histogram chart and a Blurriness Histogram chart to provide a data-centric analysis.
"""

analysis_goal = """
- Find out if the Image are diverse enough
- Explain potential impact on model performance on different data slices.
- Recommend Data augmentation techniques.
"""

prompt = PromptGenerator().generate(
    analysis_type=analysis_type, analysis_goal=analysis_goal
)

logger = logging.getLogger(__name__)
per_chart_analysis_agent_quality = Agent(model=report_model, system_prompt=prompt)


def analyze_images_quality(pctx: PContext) -> list[ReportContent]:
    df = pctx.get(cols=["asset_id", "blur_score", "contrast", "luminance"])

    logger.info("📊 Generating image quality charts...")
    chart_reports = generate_image_quality_charts(df)

    logger.info("📌 Detecting image quality outliers...")
    outlier_reports = generate_image_quality_outliers(df)

    all_reports = chart_reports + outlier_reports

    logger.info("🧠 Syncing all report content...")
    sync_reports(pctx, all_reports)

    return all_reports


def generate_image_quality_charts(
    df: pd.DataFrame,
) -> list[ReportContent | ReportError]:
    try:
        stats = ImageQualityStats(df).compute()

        chart_interpreter = MultiChartStatsInterpreter(
            stats=stats,
            agent=per_chart_analysis_agent_quality,
            prompt=prompt,
        )
    except Exception as e:
        logger.warning(f"Failed to compute image quality stats: {e}")
        return [
            ReportError(
                section=ReportContentName.BLUR_ISSUES.section,
                message="Failed to compute image quality stats",
            )
        ]

    return [
        chart_interpreter.interpret_group_to_content(
            group_name="image_statistics",
            section=ReportContentName.IMAGE_STATISTICS.section,
            sub_section=ReportContentName.IMAGE_STATISTICS.sub_section,
            name=ReportContentName.IMAGE_STATISTICS.content,
        )
    ]


def generate_image_quality_outliers(
    df: pd.DataFrame,
) -> list[ReportContent | ReportError]:
    try:
        stats = ImageQualityOutlierStats(df).compute()
        interpreter = OutlierInterpreter(stats)
    except Exception as e:
        logger.warning(f"Failed to compute image quality outliers: {e}")
        return [
            ReportError(
                section=ReportContentName.BLUR_ISSUES.section,
                message="Failed to compute image quality outliers",
            )
        ]
    return [
        interpreter.format_group_content(
            group_name="blur",
            section=ReportContentName.BLUR_ISSUES.section,
            sub_section=ReportContentName.BLUR_ISSUES.sub_section,
            name=ReportContentName.BLUR_ISSUES.content,
        ),
        interpreter.format_group_content(
            group_name="contrast",
            section=ReportContentName.CONTRAST_ISSUES.section,
            sub_section=ReportContentName.CONTRAST_ISSUES.sub_section,
            name=ReportContentName.CONTRAST_ISSUES.content,
        ),
        interpreter.format_group_content(
            group_name="luminance",
            section=ReportContentName.LUMINANCE_ISSUES.section,
            sub_section=ReportContentName.LUMINANCE_ISSUES.sub_section,
            name=ReportContentName.LUMINANCE_ISSUES.content,
        ),
    ]
