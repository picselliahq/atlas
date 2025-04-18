from agents.common.interpreter.analysis.base_interpreter import BaseInterpreter
from agents.common.models.actions import PossibleActions
from agents.common.models.contents import Assets, ReportContent
from agents.common.utils import data_reference_tag
from agents.image_agent.analysis.clip.stats import ClipStats


class OutlierInterpreter(BaseInterpreter[ClipStats, ReportContent]):
    def interpret(self):
        pass

    def format(
        self, section: str, sub_section: str, content_name: str, agent_output=None
    ) -> ReportContent:
        outlier_ids = [self.stats.asset_ids[i] for i in self.stats.outlier_indices]
        chart_id = "chart-1"
        if outlier_ids:
            data = {chart_id: Assets(type="asset-list", ids=outlier_ids)}
            text = (
                f"The following images were detected as outliers based on their CLIP embedding distance "
                f"from the dataset’s global center: {data_reference_tag(chart_id)}"
            )
            return ReportContent(
                text=text,
                data=data,
                section=section,
                sub_section=sub_section,
                name=content_name,
                potential_actions=[PossibleActions.TAG, PossibleActions.DELETE],
            )

        return ReportContent(
            text="No outlier images were detected based on CLIP embeddings.",
            data={},
            section=section,
            sub_section=sub_section,
            name=content_name,
            potential_actions=[],
        )


class DuplicateInterpreter(BaseInterpreter[ClipStats, ReportContent]):
    def interpret(self):
        pass

    def format(
        self, section: str, sub_section: str, content_name: str, agent_output=None
    ) -> ReportContent:
        duplicate_data = {}
        text_lines = []
        for idx, cluster in enumerate(self.stats.duplicate_clusters.values()):
            duplicate_ids = [self.stats.asset_ids[i] for i in cluster]
            if not duplicate_ids:
                continue
            chart_id = f"chart-{idx + 1}"
            duplicate_data[chart_id] = Assets(type="asset-list", ids=duplicate_ids)
            text_lines.append(f"\n 🔹 Near-duplicate cluster {idx + 1}:")
            text_lines.append(f" {data_reference_tag(chart_id)}")

        if duplicate_data:
            text = (
                "Several groups of visually similar images were found and may indicate duplicates. "
                "These clusters were identified using UMAP and DBSCAN:\n"
                + "\n".join(text_lines)
            )
            return ReportContent(
                text=text,
                data=duplicate_data,
                section=section,
                sub_section=sub_section,
                name=content_name,
                potential_actions=[PossibleActions.CLEAN, PossibleActions.DELETE],
            )

        return ReportContent(
            text="No duplicate images were identified in the dataset.",
            data={},
            section=section,
            sub_section=sub_section,
            name=content_name,
            potential_actions=[],
        )
