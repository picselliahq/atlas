import numpy as np
import pandas as pd

from agents.common.stats.outlier.base_outlier_stats import BaseOutlierStats


class IntraClassEmbeddingCentroidOutliers(BaseOutlierStats):
    def __init__(
        self,
        df: pd.DataFrame,
        shape_distances: pd.DataFrame,
        outlier_percentile: float = 95.0,
        min_samples_per_class: int = 6,
    ):
        super().__init__(df)
        self.shape_distances = shape_distances
        self.outlier_percentile = outlier_percentile
        self.min_samples_per_class = min_samples_per_class

    def _prepare_data(self):
        pass

    def _compute_outliers(self):
        elements = []

        for label, group in self.shape_distances.groupby("label"):
            if len(group) < self.min_samples_per_class:
                continue

            threshold = np.percentile(
                group["distance_to_centroid"], self.outlier_percentile
            )
            outliers = group[group["distance_to_centroid"] > threshold]

            if not outliers.empty:
                elements.append(
                    {
                        "label": label,
                        "ids": outliers["shape_id"].tolist(),
                    }
                )

        self.outlier_groups["centroid_distance_outliers"] = {
            "shapes": {
                "description": f"Shapes with embeddings above the {self.outlier_percentile}th percentile of distance to class centroid",
                "elements": elements,
            }
        }
