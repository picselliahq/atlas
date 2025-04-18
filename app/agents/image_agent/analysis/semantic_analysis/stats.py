import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from agents.common.stats.base_stats import BaseStats


class SemanticCaptionStats(BaseStats["SemanticCaptionStats"]):
    def __init__(
        self,
        df: pd.DataFrame,
        eps: float = 0.5,
        min_samples: int = 2,
        top_words: int = 10,
    ):
        super().__init__(df)
        self.df: pd.DataFrame = self.df[self.df["caption"].notnull()]

        self.eps = eps
        self.min_samples = min_samples
        self.top_words = top_words

        self._vectorizer = TfidfVectorizer(stop_words="english")
        self.clustered_df = None
        self.cluster_caption_summary: dict[int, dict] = {}

    def compute(
        self,
    ) -> "SemanticCaptionStats":
        self._cluster_captions()
        self._summarize_clusters()
        return self

    def _cluster_captions(
        self,
    ):
        X = self._vectorizer.fit_transform(self.df["caption"])
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.df["semantic_cluster"] = dbscan.fit_predict(X)
        self.clustered_df = self.df

    def _summarize_clusters(self):
        grouped = self.df.groupby("semantic_cluster")
        self.cluster_caption_summary = {
            cluster_id: {
                "captions": self._generate_average_caption(group["caption"].tolist())
            }
            for cluster_id, group in grouped
        }

    def _generate_average_caption(self, captions: list[str]) -> str:
        tfidf_matrix = self._vectorizer.fit_transform(captions)
        feature_names = self._vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1
        word_scores = dict(zip(feature_names, tfidf_scores, strict=False))
        top_words_list = [
            word
            for word, _ in sorted(
                word_scores.items(), key=lambda x: x[1], reverse=True
            )[: self.top_words]
        ]
        return " ".join(str(word) for word in top_words_list)
