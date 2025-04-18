import logging

from pydantic_ai import Agent

from agents.ai_models import report_model
from agents.annotation_agent.analysis.tightness.interpreter import TightnessInterpreter
from agents.annotation_agent.analysis.tightness.stats import TightnessStats
from agents.annotation_agent.common.get_context import PContext
from agents.common.interpreter.merged_report_builder import MergedReportBuilder
from agents.common.interpreter.multi_chart_stats_interpreter import (
    MultiChartStatsInterpreter,
)
from agents.common.models.contents import ReportContent
from agents.prompts.generator import PromptGenerator
from services.report_enums import SubSectionName

per_chart_prompt_tightness = """
You're a helpful computer vision assistant for beginners working on object detection.

🔍 Your Role:
Analyze the box plots of bounding box deltas (Δx1, Δy1, Δx2, Δy2) per object class (e.g., car, girafe) that I provide. These charts compare the annotated bounding boxes to the suggested "corrected" boxes.

🎯 Who You're Helping:

Model builders (data scientists, ML engineers): help them understand how label quality could affect training.
Labeling teams (annotators, QA reviewers): help them spot and fix inconsistencies or bias in box placement.

📋 What to Include:

✅ Recommendations First:

List any classes with issues
For each class, give:
What the labeling team should fix (e.g., consistent box tightness, reducing looseness, etc.)
What the model team should watch out for (e.g., box shift bias, outliers, label imbalance)

📊 Interpret the Box Plots:

Look at Δx1 / Δy1 for top-left shifts, and Δx2 / Δy2 for bottom-right shifts
Point out any bias (e.g., boxes are always looser in one direction)
Quantify medians and spread (e.g., "median Δx1 for girafe is -0.07")
Highlight high variance or skew that might hurt model learning

🧠 Be Beginner-Friendly:
Use short bullets
Avoid heavy jargon
Call out exact problems and numbers clearly

🛑 Flag:

Systematic looseness or tightness in boxes
Inconsistent annotation styles across classes
If a class appears to be more noisy or inconsistent than others

"""

analysis_type = """
Analyze the box plots of bounding box deltas (Δx1, Δy1, Δx2, Δy2) per object class
"""

analysis_goal = """
Look at Δx1 / Δy1 for top-left shifts, and Δx2 / Δy2 for bottom-right shifts
Point out any bias (e.g., boxes are always looser in one direction)
Quantify medians and spread (e.g., "median Δx1 for girafe is -0.07")
Highlight high variance or skew that might hurt model learning
"""

prompt = PromptGenerator().generate(
    analysis_type=analysis_type, analysis_goal=analysis_goal
)

logger = logging.getLogger(__name__)
per_chart_analysis_agent_tightness = Agent(
    model=report_model,
    system_prompt=prompt,
)


def find_tightness_issues(pctx: PContext) -> list[ReportContent]:
    df = pctx.get(["shape_id", "shape", "x", "y", "w", "h", "label"])

    if "x" not in df.columns or "y" not in df.columns:
        logger.info("⚠️ Missing 'x' or 'y' columns. Skipping tightness stats analysis.")
        return []

    logger.info("📊 Running tightness stats analysis...")
    unique_ids = df["asset_id"].unique()[:5]
    target_ids: list[str] = [str(uid) for uid in unique_ids]

    logger.info(f"🔍 Running SAM on {len(target_ids)} images...")
    stats = TightnessStats(df=df, dataset=pctx.dataset, asset_ids=target_ids).compute()

    logger.info("🧠 Running LLM-based tightness suggestion analysis...")
    tightness_suggestions_contents = TightnessInterpreter(
        stats=stats, min_treshold=0.5, max_treshold=0.9
    ).run(
        section=SubSectionName.TIGHTNESS.section.value,
        sub_section=SubSectionName.TIGHTNESS.name,
    )

    logger.info("🧠 Running LLM interpretation for tightness charts...")
    chart_content = MultiChartStatsInterpreter(
        stats=stats,
        agent=per_chart_analysis_agent_tightness,
        prompt=prompt,
    ).interpret_group_to_content(
        group_name="tightness",
        section=SubSectionName.TIGHTNESS.section.value,
        sub_section=SubSectionName.TIGHTNESS.name,
        name="Tightness Box Plots Analysis",
    )

    logger.info("🧬 Merging charts + shape suggestions into single report...")
    merged_content = MergedReportBuilder(
        [chart_content] + tightness_suggestions_contents,
        title="Full Tightness Analysis",
    ).build()

    pctx.context_service.sync_content(merged_content)
    return [merged_content]
