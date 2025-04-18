import pandas as pd

from agents.common.stats.outlier.base_outlier_stats import (
    AssetOutlierGroupItem,
    BaseOutlierStats,
    ShapeOutlierGroupItem,
)


class ShapeOutlierStats(BaseOutlierStats):
    def __init__(
        self,
        df: pd.DataFrame,
        shape_stats: dict[str, dict[str, float]],
        z_thresh: float = 2.5,
    ):
        super().__init__(df)
        self.stats = shape_stats
        self.z_thresh = z_thresh

    def _prepare_data(self) -> None:
        pass

    def _compute_outliers(self) -> None:
        self.outlier_groups["area"] = self.find_area_outliers()
        self.outlier_groups["aspect_ratio"] = self.find_aspect_ratio_outliers()
        self.outlier_groups["density"] = self.find_density_outliers()

    def _combine_outliers(
        self, z_outliers: pd.DataFrame, iqr_outliers: pd.DataFrame
    ) -> pd.DataFrame:
        return pd.concat([z_outliers, iqr_outliers]).drop_duplicates(
            subset=["shape_id"]
        )

    def find_area_outliers(self) -> dict[str, ShapeOutlierGroupItem]:
        small_elements = []
        large_elements = []
        typical_elements = []

        for label, group in self.df.groupby("label"):
            z_low, z_high = pd.DataFrame(), pd.DataFrame()
            stats = self.stats.get(label)

            # --- Z-score
            if stats and stats.get("std_area", 0) != 0:
                mean = stats["mean_area"]
                std = stats["std_area"]
                z_scores = (group["area"] - mean) / std
                z_low = group[z_scores < -self.z_thresh]
                z_high = group[z_scores > self.z_thresh]

            # --- IQR
            if group.shape[0] >= 5:
                q1 = group["area"].quantile(0.25)
                q3 = group["area"].quantile(0.75)
                iqr = q3 - q1
                iqr_low = group[group["area"] < q1 - 1.5 * iqr]
                iqr_high = group[group["area"] > q3 + 1.5 * iqr]
            else:
                iqr_low, iqr_high = pd.DataFrame(), pd.DataFrame()

            low = self._combine_outliers(z_low, iqr_low)
            high = self._combine_outliers(z_high, iqr_high)
            outlier_ids = set(low["shape_id"]).union(high["shape_id"])

            if not low.empty:
                small_elements.append(
                    {"label": label, "ids": low["shape_id"].astype(str).tolist()}
                )

            if not high.empty:
                large_elements.append(
                    {"label": label, "ids": high["shape_id"].astype(str).tolist()}
                )

            if outlier_ids:
                sample = self._get_inlier_sample(group, outlier_ids, column="area", n=1)
                if not sample.empty:
                    typical_elements.append(
                        {"label": label, "ids": sample["shape_id"].astype(str).tolist()}
                    )

        return {
            "small_shapes": {
                "description": "Shapes that are unusually small for their label",
                "elements": small_elements,
            },
            "large_shapes": {
                "description": "Shapes that are unusually large for their label",
                "elements": large_elements,
            },
            "typical_shapes": {
                "description": "Shapes with typical area across labels",
                "elements": typical_elements,
            },
        }

    def find_aspect_ratio_outliers(self) -> dict[str, ShapeOutlierGroupItem]:
        extreme_elements = []
        typical_elements = []

        for label, group in self.df.groupby("label"):
            if group.shape[0] < 2:
                continue

            # --- Z-score
            mean = group["aspect_ratio"].mean()
            std = group["aspect_ratio"].std()
            z_outliers = (
                group[((group["aspect_ratio"] - mean) / std).abs() > self.z_thresh]
                if std > 0
                else pd.DataFrame()
            )

            # --- IQR
            if group.shape[0] >= 5:
                q1 = group["aspect_ratio"].quantile(0.25)
                q3 = group["aspect_ratio"].quantile(0.75)
                iqr = q3 - q1
                iqr_outliers = group[
                    (group["aspect_ratio"] < q1 - 1.5 * iqr)
                    | (group["aspect_ratio"] > q3 + 1.5 * iqr)
                ]
            else:
                iqr_outliers = pd.DataFrame()

            outliers = self._combine_outliers(z_outliers, iqr_outliers)
            outlier_ids = set(outliers["shape_id"])

            if not outliers.empty:
                extreme_elements.append(
                    {"label": label, "ids": outliers["shape_id"].astype(str).tolist()}
                )

            if outlier_ids:
                sample = self._get_inlier_sample(
                    group, outlier_ids, column="aspect_ratio", n=1
                )
                if not sample.empty:
                    typical_elements.append(
                        {"label": label, "ids": sample["shape_id"].astype(str).tolist()}
                    )

        return {
            "extreme_ratios": {
                "description": "Shapes with extreme aspect ratios (too flat or too tall)",
                "elements": extreme_elements,
            },
            "typical_ratios": {
                "description": "Shapes with typical aspect ratios across labels",
                "elements": typical_elements,
            },
        }

    def find_density_outliers(
        self, sample_n: int = 5, use_quantiles: bool = True
    ) -> dict[str, AssetOutlierGroupItem]:
        shape_counts = self.df.groupby("asset_id")["shape_id"].count()

        outlier_ids = set()
        group_items = {}

        if use_quantiles:
            q1 = shape_counts.quantile(0.05)
            q3 = shape_counts.quantile(0.95)

            low = shape_counts[shape_counts <= q1].index.astype(str).tolist()
            high = shape_counts[shape_counts >= q3].index.astype(str).tolist()
        else:
            mean = shape_counts.mean()
            std = shape_counts.std()
            if std == 0:
                return {}

            z_scores = (shape_counts - mean) / std
            low = z_scores[z_scores < -self.z_thresh].index.astype(str).tolist()
            high = z_scores[z_scores > self.z_thresh].index.astype(str).tolist()

        outlier_ids.update(low + high)

        if low:
            group_items["few_objects"] = {
                "description": "Images with very few objects (underpopulated)",
                "ids": low,
            }

        if high:
            group_items["many_objects"] = {
                "description": "Images with too many objects (overcrowded)",
                "ids": high,
            }

        if outlier_ids:
            inlier_ids = shape_counts.index.difference(outlier_ids)
            if not inlier_ids.empty:
                inliers = shape_counts.loc[inlier_ids].copy()
                median_val = shape_counts.median()
                inliers = inliers.to_frame("shape_count")
                inliers["distance_to_median"] = (
                    inliers["shape_count"] - median_val
                ).abs()
                sample = (
                    inliers.sort_values("distance_to_median")
                    .head(sample_n)
                    .index.astype(str)
                    .tolist()
                )
                if sample:
                    group_items["typical_density"] = {
                        "description": "Images with typical object count",
                        "ids": sample,
                    }

        return group_items

    def _get_inlier_sample(
        self, group: pd.DataFrame, outlier_ids: set, column: str, n: int = 5
    ) -> pd.DataFrame:
        inliers = group[~group["shape_id"].isin(outlier_ids)].copy()
        median_val = inliers[column].median()
        inliers["distance_to_median"] = (inliers[column] - median_val).abs()
        return inliers.sort_values("distance_to_median").head(n)
