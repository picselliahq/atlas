import logging

import pandas as pd
from pydantic_ai import Agent

from agents.ai_models import report_model
from agents.annotation_agent.analysis.shape_analysis.outliers import ShapeOutlierStats
from agents.annotation_agent.analysis.shape_analysis.stats import ShapeStats
from agents.annotation_agent.common.get_context import PContext
from agents.common.interpreter.analysis.multi_chart_stats_interpreter import (
    MultiChartStatsInterpreter,
)
from agents.common.interpreter.outlier.outlier_interpreter import OutlierInterpreter
from agents.common.models.contents import ReportContent
from agents.common.utils import sync_reports
from agents.prompts.generator import PromptGenerator
from services.report_enums import ReportContentName

analysis_type = """
- Shape Density: How many objects are labeled per image
- Aspect Ratio per Class: Shape of boxes (width vs height)
- Object Area per Class: How big each class typically is
- Class Distribution: How often each class is used
- Object Area Distribution: Overall box size spread (in pixels)
"""
analysis_goal = """
- Overall aspect of objects and is it matching common-sense
- Size of objects - is SAHI a technique to consider? (only when analysing the size of the object)
- Outliers in size and shape
- Analyse against common sense.
"""

prompt = PromptGenerator().generate(
    analysis_type=analysis_type, analysis_goal=analysis_goal
)
logger = logging.getLogger(__name__)
per_chart_analysis_agent_shapes = Agent(
    model=report_model,
    system_prompt=prompt,
)


def analyze_object_shapes(pctx: PContext) -> list[ReportContent]:
    df = pctx.get(
        cols=["label", "shape_id", "x", "y", "w", "h", "image_width", "image_height"]
    )

    if df.empty or "x" not in df.columns or "y" not in df.columns:
        logger.warning("⚠️ Skipping shape analysis due to missing spatial columns.")
        return []

    logger.info("📊 Generating shape-related chart reports...")
    chart_reports = generate_shape_charts(df)

    logger.info("📌 Detecting shape-related outliers...")
    outlier_reports = generate_shape_outliers(df)

    all_reports = chart_reports + outlier_reports
    logger.info("🧠 Syncing all shape reports...")
    sync_reports(pctx, all_reports)

    return all_reports


def generate_shape_charts(df: pd.DataFrame) -> list[ReportContent]:
    stats = ShapeStats(df).compute()

    interpreter = MultiChartStatsInterpreter(
        stats=stats,
        agent=per_chart_analysis_agent_shapes,
        prompt=prompt,
    )

    reports = [
        interpreter.interpret_group_to_content(
            group_name="class_frequency",
            section=ReportContentName.CLASS_DISTRIBUTION.section,
            sub_section=ReportContentName.CLASS_DISTRIBUTION.sub_section,
            name=ReportContentName.CLASS_DISTRIBUTION.content,
        )
    ]

    if "object_analysis" in stats.chart_groups:
        reports.append(
            interpreter.interpret_group_to_content(
                group_name="object_analysis",
                section=ReportContentName.SINGLE_OBJECT_ANALYSIS.section,
                sub_section=ReportContentName.SINGLE_OBJECT_ANALYSIS.sub_section,
                name=ReportContentName.SINGLE_OBJECT_ANALYSIS.content,
            )
        )

    return reports


def generate_shape_outliers(df: pd.DataFrame) -> list[ReportContent]:
    stats = ShapeStats(df).compute()
    outlier_stats = ShapeOutlierStats(
        df=stats.df, shape_stats=stats.per_class_stats, z_thresh=2.5
    ).compute()

    interpreter = OutlierInterpreter(outlier_stats)

    return [
        interpreter.format_group_content(
            group_name="area",
            section=ReportContentName.OUTLIER_SHAPE_AREA.section,
            sub_section=ReportContentName.OUTLIER_SHAPE_AREA.sub_section,
            name=ReportContentName.OUTLIER_SHAPE_AREA.content,
        ),
        interpreter.format_group_content(
            group_name="aspect_ratio",
            section=ReportContentName.OUTLIER_ASPECT_RATIO.section,
            sub_section=ReportContentName.OUTLIER_ASPECT_RATIO.sub_section,
            name=ReportContentName.OUTLIER_ASPECT_RATIO.content,
        ),
        interpreter.format_group_content(
            group_name="density",
            section=ReportContentName.OUTLIER_DENSITY.section,
            sub_section=ReportContentName.OUTLIER_DENSITY.sub_section,
            name=ReportContentName.OUTLIER_DENSITY.content,
        ),
    ]
