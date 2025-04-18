import pandas as pd

from agents.common.stats.outlier.base_outlier_stats import (
    AssetOutlierGroupItem,
    BaseOutlierStats,
)


class ImageQualityOutlierStats(BaseOutlierStats):
    def __init__(
        self,
        df: pd.DataFrame,
        quantile_low: float = 0.05,
        quantile_high: float = 0.95,
        blur_quantile: float = 0.99,
    ):
        super().__init__(df)
        self.q_low = quantile_low
        self.q_high = quantile_high
        self.blur_q = blur_quantile

    def _prepare_data(self) -> None:
        pass

    def _compute_outliers(self) -> None:
        if "luminance" in self.df:
            self.outlier_groups["luminance"] = self._iqr_outliers("luminance")

        if "contrast" in self.df:
            self.outlier_groups["contrast"] = self._iqr_outliers("contrast")

        if "blur_score" in self.df:
            self.outlier_groups["blur"] = self._blur_outliers()

    def _iqr_outliers(
        self, column: str, sample_n: int = 5
    ) -> dict[str, AssetOutlierGroupItem]:
        q1, q3 = self.df[column].quantile([self.q_low, self.q_high])
        iqr = q3 - q1
        lower = q1 - 2 * iqr
        upper = q3 + 2 * iqr

        low_df = self.df[self.df[column] < lower]
        high_df = self.df[self.df[column] > upper]
        inlier_df = self.df[
            (self.df[column] >= lower) & (self.df[column] <= upper)
        ].copy()

        low_ids = low_df["asset_id"].tolist()
        high_ids = high_df["asset_id"].tolist()

        if not low_ids:
            min_id = self.df.loc[self.df[column].idxmin()]["asset_id"]
            low_ids = [min_id]
            inlier_df = inlier_df[inlier_df["asset_id"] != min_id]

        if not high_ids:
            max_id = self.df.loc[self.df[column].idxmax()]["asset_id"]
            high_ids = [max_id]
            inlier_df = inlier_df[inlier_df["asset_id"] != max_id]

        median_val = self.df[column].median()
        inlier_df["distance_to_median"] = (inlier_df[column] - median_val).abs()
        sampled_inliers = (
            inlier_df.sort_values("distance_to_median")
            .head(sample_n)["asset_id"]
            .tolist()
        )

        return {
            "low_outliers": {
                "description": f"Images with very low {column}",
                "ids": low_ids,
            },
            "high_outliers": {
                "description": f"Images with very high {column}",
                "ids": high_ids,
            },
            "inliers_sample": {
                "description": f"Representative images with typical {column}",
                "ids": sampled_inliers,
            },
        }

    def _blur_outliers(self, sample_n: int = 5) -> dict[str, AssetOutlierGroupItem]:
        threshold = self.df["blur_score"].quantile(self.blur_q)
        high_df = self.df[self.df["blur_score"] >= threshold]
        inlier_df = self.df[self.df["blur_score"] < threshold].copy()

        high_ids = high_df["asset_id"].tolist()

        if inlier_df.empty:
            return {
                "high_outliers": {
                    "description": "Images with high blur score",
                    "ids": high_ids,
                },
                "inliers_sample": {
                    "description": "No representative inliers found",
                    "ids": [],
                },
            }

        median_val = self.df["blur_score"].median()
        inlier_df["distance_to_median"] = (inlier_df["blur_score"] - median_val).abs()
        sampled_inliers = (
            inlier_df.sort_values("distance_to_median")
            .head(sample_n)["asset_id"]
            .tolist()
        )

        return {
            "high_outliers": {
                "description": "Images with high blur score",
                "ids": high_ids,
            },
            "inliers_sample": {
                "description": "Representative images with typical blur",
                "ids": sampled_inliers,
            },
        }
