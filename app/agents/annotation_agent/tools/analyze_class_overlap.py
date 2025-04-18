import logging

import pandas as pd
from pydantic_ai import Agent

from agents.ai_models import report_model
from agents.annotation_agent.analysis.inter_class.outliers import InterClassOutlierStats
from agents.annotation_agent.analysis.inter_class.stats import InterClassStats
from agents.annotation_agent.common.get_context import PContext
from agents.common.interpreter.analysis.multi_chart_stats_interpreter import (
    MultiChartStatsInterpreter,
)
from agents.common.interpreter.outlier.outlier_interpreter import OutlierInterpreter
from agents.common.models.contents import ReportContent
from agents.common.utils import sync_reports
from agents.prompts.generator import PromptGenerator
from services.report_enums import ReportContentName

logger = logging.getLogger(__name__)
prompt = PromptGenerator().generate(
    analysis_type="class co-occurrence matrix or a class overlap matrix from an image training dataset.",
    analysis_goal="""
For co-occurence:
- Common class pairings
- Class imbalance
- Label redundancy (e.g., people vs pedestrian)
- Rare or isolated classes

For overlap matrix:
- Physical interaction between classes
- Potential label confusion due to overlapping region
- High-density class clusters
""",
)

per_chart_analysis_agent_overlap = Agent(
    model=report_model,
    system_prompt=prompt,
)


def analyze_class_overlap(pctx: PContext) -> list[ReportContent]:
    df = pctx.get(
        cols=[
            "label",
            "w",
            "h",
            "image_width",
            "image_height",
            "x",
            "y",
            "shape_id",
            "label_id",
        ]
    )

    if "x" not in df.columns or "y" not in df.columns:
        logger.warning("⚠️ Missing 'x' or 'y' columns. Skipping inter class analysis.")
        return []

    if df["label"].nunique() <= 1:
        logger.warning("⚠️ Only one label class found. Skipping inter class analysis.")
        return []

    logger.info("📊 Generating inter-class stats...")
    chart_stats, chart_reports = generate_interclass_charts(df)

    logger.info("📌 Detecting inter-class outliers...")
    outlier_reports = generate_interclass_outliers(df, chart_stats)

    all_reports = chart_reports + outlier_reports

    logger.info("🧠 Syncing all report content...")
    sync_reports(pctx, all_reports)

    return all_reports


def generate_interclass_charts(df: pd.DataFrame) -> tuple[dict, list[ReportContent]]:
    chart_stats = InterClassStats(df=df).compute()

    interpreter = MultiChartStatsInterpreter(
        stats=chart_stats,
        agent=per_chart_analysis_agent_overlap,
        prompt=prompt,
    )

    return chart_stats.stats, [
        interpreter.interpret_group_to_content(
            group_name="cooccurrence",
            section=ReportContentName.TOP_COOCCURRENCES.section,
            sub_section=ReportContentName.TOP_COOCCURRENCES.sub_section,
            name=ReportContentName.TOP_COOCCURRENCES.content,
        ),
        interpreter.interpret_group_to_content(
            group_name="overlap",
            section=ReportContentName.TOP_OVERLAPS.section,
            sub_section=ReportContentName.TOP_OVERLAPS.sub_section,
            name=ReportContentName.TOP_OVERLAPS.content,
        ),
    ]


def generate_interclass_outliers(
    df: pd.DataFrame, chart_stats: dict
) -> list[ReportContent]:
    stats = InterClassOutlierStats(
        df=df,
        cooccurrence_matrix=chart_stats["cooccurrence_matrix"],
        overlap_matrix=chart_stats["overlap_count_matrix"],
    ).compute()

    interpreter = OutlierInterpreter(stats)

    return [
        interpreter.format_group_content(
            group_name="missing_expected_cooccurrence",
            section=ReportContentName.MISSING_COOCCURRENCE.section,
            sub_section=ReportContentName.MISSING_COOCCURRENCE.sub_section,
            name=ReportContentName.MISSING_COOCCURRENCE.content,
        ),
        interpreter.format_group_content(
            group_name="unexpected_cooccurrence",
            section=ReportContentName.UNEXPECTED_COOCCURRENCE.section,
            sub_section=ReportContentName.UNEXPECTED_COOCCURRENCE.sub_section,
            name=ReportContentName.UNEXPECTED_COOCCURRENCE.content,
        ),
        interpreter.format_group_content(
            group_name="unexpected_overlap",
            section=ReportContentName.UNEXPECTED_OVERLAP.section,
            sub_section=ReportContentName.UNEXPECTED_OVERLAP.sub_section,
            name=ReportContentName.UNEXPECTED_OVERLAP.content,
        ),
    ]
