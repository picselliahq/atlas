import logging

from agents.common.models.assets import AssetComparison, AssetComparisonElement, Assets
from agents.common.models.contents import ReportContent, ReportError
from agents.common.models.shapes import Shapes
from agents.common.stats.outlier.base_outlier_stats import (
    AssetOutlierGroupItem,
    BaseOutlierStats,
    ComparisonOutlierGroupItem,
    ShapeOutlierGroupItem,
)
from agents.common.utils import data_reference_tag

MAX_PREVIEW_ITEMS = 100

logger = logging.getLogger(__name__)


class OutlierInterpreter:
    def __init__(self, outlier_stats: BaseOutlierStats):
        self.outlier_groups = outlier_stats.outlier_groups

    def format_group_content(
        self,
        group_name: str,
        section: str,
        sub_section: str,
        name: str,
        actions: list[str] | None = None,
    ) -> ReportContent | ReportError:
        try:
            group_items = self.outlier_groups.get(group_name, {})
            if not group_items:
                return ReportContent(
                    text=f"No data found for group `{group_name}`.",
                    data={},
                    section=section,
                    sub_section=sub_section,
                    name=name,
                    potential_actions=[],
                )

            lines: list[str] = []
            data: dict = {}
            chart_idx = 1

            for _key, item in group_items.items():
                if "ids" in item:
                    chart_idx = self._render_asset_outlier_item(
                        item=item,
                        lines=lines,
                        data=data,
                        chart_idx=chart_idx,
                    )
                elif "pairs" in item:
                    chart_idx = self._render_comparison_outlier_item(
                        item=item,
                        lines=lines,
                        data=data,
                        chart_idx=chart_idx,
                    )
                elif "elements" in item:
                    chart_idx = self._render_shape_group_item(
                        item=item,
                        lines=lines,
                        data=data,
                        chart_idx=chart_idx,
                    )

            return ReportContent(
                text="\n".join(lines),
                data=data,
                section=section,
                sub_section=sub_section,
                name=name,
                potential_actions=actions or [],
            )
        except Exception as e:
            logger.warning(f"Fail to format group: {e}")
            return ReportError(
                section=section,
                sub_section=sub_section,
                message=f"Failed to format group {group_name}.",
            )

    def _render_asset_outlier_item(
        self,
        item: AssetOutlierGroupItem,
        lines: list[str],
        data: dict,
        chart_idx: int,
    ) -> int:
        ids = item.get("ids", [])[:MAX_PREVIEW_ITEMS]
        desc = item.get("description", "")

        if not ids or not desc:
            return chart_idx

        chart_id = f"chart-{chart_idx}"
        data[chart_id] = Assets(type="asset-list", ids=ids)
        lines.append(f"🔹 {desc}:")
        lines.append(f"{data_reference_tag(chart_id)}")
        return chart_idx + 1

    def _render_comparison_outlier_item(
        self,
        item: ComparisonOutlierGroupItem,
        lines: list[str],
        data: dict,
        chart_idx: int,
    ) -> int:
        desc = item.get("description", "")
        pairs = item.get("pairs", [])

        if desc:
            lines.append(f"🔹 {desc}:")

        if not pairs:
            lines.append("\n No label pairs with relevant outliers were found. \n")
            return chart_idx

        for pair in pairs:
            label_1 = pair["label_1"].capitalize()
            label_2 = pair["label_2"].capitalize()
            elements: list[AssetComparisonElement] = pair["elements"][
                :MAX_PREVIEW_ITEMS
            ]

            chart_id = f"chart-{chart_idx}"
            data[chart_id] = AssetComparison(type="asset-comparisons", data=elements)
            lines.append(f"\n {label_1} and {label_2}: \n")
            lines.append(f"{data_reference_tag(chart_id)}")
            chart_idx += 1

        return chart_idx

    def _render_shape_group_item(
        self,
        item: ShapeOutlierGroupItem,
        lines: list[str],
        data: dict,
        chart_idx: int,
    ) -> int:
        desc = item.get("description", "")
        elements = item.get("elements", [])

        if desc:
            lines.append(f"🔹 {desc}:")

        if not elements:
            lines.append("\n No shape outliers found in this group. \n")
            return chart_idx

        all_ids = []
        for el in elements:
            ids = el.get("ids", [])
            if not ids:
                continue
            all_ids.extend(ids)
        all_ids = all_ids[:MAX_PREVIEW_ITEMS]
        chart_id = f"chart-{chart_idx}"
        data[chart_id] = Shapes(type="shape-list", ids=all_ids)
        lines.append(f"{data_reference_tag(chart_id)}")
        chart_idx += 1

        return chart_idx

    def list_groups(self) -> list[str]:
        return list(self.outlier_groups.keys())
