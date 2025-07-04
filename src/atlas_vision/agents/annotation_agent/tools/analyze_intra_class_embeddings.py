import logging

from atlas_vision.agents.annotation_agent.analysis.intra_class.outliers import (
    IntraClassEmbeddingCentroidOutliers,
)
from atlas_vision.agents.annotation_agent.analysis.intra_class.stats import (
    IntraClassEmbeddingCentroidStats,
)
from atlas_vision.agents.annotation_agent.common.get_context import PContext
from atlas_vision.agents.common.interpreter.outlier.outlier_interpreter import (
    OutlierInterpreter,
)
from atlas_vision.agents.common.models.contents import ReportContent
from atlas_vision.agents.common.utils import sync_reports
from atlas_vision.services.report_enums import ReportContentName

logger = logging.getLogger(__name__)


def analyze_intra_class_embeddings(pctx: PContext) -> list[ReportContent]:
    df = pctx.get(cols=["label", "shape_id", "shape_embeddings"])

    if "shape_embeddings" not in df.columns or df["label"].nunique() <= 1:
        logger.warning(
            "⚠️ Not enough data or shape embeddings missing. Skipping intra-class analysis."
        )
        return []

    reports: list[ReportContent] = []

    logger.info("📊Centroid stats...")
    centroid_stats = IntraClassEmbeddingCentroidStats(df=df).compute()

    logger.info("📌 Outliers by distance to centroid...")
    centroid_outliers = IntraClassEmbeddingCentroidOutliers(
        df=df,
        shape_distances=centroid_stats.stats["shape_distances"],
        outlier_percentile=95.0,
    ).compute()

    interpreter_centroid = OutlierInterpreter(centroid_outliers)
    report_centroid = interpreter_centroid.format_group_content(
        group_name="centroid_distance_outliers",
        section=ReportContentName.INTRA_CLASS_CENTROID_OUTLIERS.section,
        sub_section=ReportContentName.INTRA_CLASS_CENTROID_OUTLIERS.sub_section,
        name="Intra-Class Centroid Distance Outliers",
    )
    reports.append(report_centroid)

    logger.info("🧠 Syncing intra-class reports...")
    sync_reports(pctx, reports)

    return reports
