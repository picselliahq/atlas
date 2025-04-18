import numpy as np
import pandas as pd
import umap
from sklearn.cluster import DBSCAN

from agents.common.stats.analysis.base_stats import BaseStats
from agents.common.utils import convert_to_list


class ClipStats(BaseStats["ClipStats"]):
    def __init__(
        self,
        df: pd.DataFrame,
        top_percent_outlier: float = 3.0,
        dbscan_eps: float = 0.005,
        dbscan_min_samples: int = 2,
    ):
        super().__init__(df)
        self.df: pd.DataFrame

        self.top_percent_outlier = top_percent_outlier
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        self.df["clip_embeddings"] = self.df["clip_embeddings"].apply(convert_to_list)
        self.df = self.df[
            self.df["clip_embeddings"].apply(
                lambda x: isinstance(x, list) and len(x) > 0
            )
        ]
        self.embeddings = np.array(self.df["clip_embeddings"].tolist())
        self.asset_ids = self.df["asset_id"].tolist()

        self.outlier_indices: list[int] = []
        self.duplicate_clusters: dict[int, list[int]] = {}

    def compute(
        self,
    ) -> "ClipStats":
        self.outlier_indices = self._find_outliers_from_embeddings()
        self.duplicate_clusters = self._find_duplicate_clusters()
        return self

    def _find_outliers_from_embeddings(self) -> list[int]:
        centroid = np.mean(self.embeddings, axis=0)
        distances = np.linalg.norm(self.embeddings - centroid, axis=1)
        threshold_index = int(len(distances) * (1 - self.top_percent_outlier / 100.0))
        sorted_indices = np.argsort(distances)
        return sorted_indices[threshold_index:].tolist()

    def _find_duplicate_clusters(self) -> dict[int, list[int]]:
        reducer = umap.UMAP(
            n_components=2, n_neighbors=15, min_dist=0.1, random_state=42
        )
        umap_embeddings = reducer.fit_transform(self.embeddings)
        labels = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        ).fit_predict(umap_embeddings)

        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            if label != -1:
                clusters.setdefault(label, []).append(idx)
        return {k: v for k, v in clusters.items() if len(v) > 1}
