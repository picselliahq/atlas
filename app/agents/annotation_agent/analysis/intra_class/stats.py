from typing import Any

import numpy as np
import pandas as pd

from agents.common.stats.analysis.base_chart_stats import BaseChartStats


class IntraClassEmbeddingCentroidStats(
    BaseChartStats["IntraClassEmbeddingCentroidStats"]
):
    def __init__(
        self,
        df: pd.DataFrame,
        embedding_col: str = "shape_embeddings",
        label_col: str = "label",
    ):
        super().__init__(df)
        self.embedding_col = embedding_col
        self.label_col = label_col
        self.stats: dict[str, Any] = {}

    def _prepare_data(self):
        self.df[self.embedding_col] = self.df[self.embedding_col].apply(
            lambda x: np.array(x) if isinstance(x, list) else x
        )

    def _generate_all_chart_groups(self) -> None:
        pass

    def _compute_stats(self):
        self._compute_centroid_distances()

    def _compute_centroid_distances(self):
        all_distances = []
        label_centroids: dict[str, np.ndarray] = {}
        shape_distances: list[dict[str, Any]] = []

        for label, group in self.df.groupby(self.label_col):
            embeddings = np.stack(group[self.embedding_col].values)
            centroid = embeddings.mean(axis=0)
            label_centroids[label] = centroid

            distances = np.linalg.norm(embeddings - centroid, axis=1)

            for i, (_, row) in enumerate(group.iterrows()):
                shape_distances.append(
                    {
                        "shape_id": row["shape_id"],
                        "label": label,
                        "distance_to_centroid": distances[i],
                    }
                )
                all_distances.append(distances[i])

        self.stats["shape_distances"] = pd.DataFrame(shape_distances)
        self.stats["label_centroids"] = label_centroids
        self.stats["global_centroid_distances"] = all_distances
