import json
from typing import Any

from pydantic_ai.agent import AgentRunResult

from agents.common.interpreter.base_interpreter import BaseInterpreter
from agents.common.models.actions import PossibleActions
from agents.common.models.contents import Assets, ReportContent
from agents.common.utils import clean_newlines, data_reference_tag
from agents.image_agent.analysis.semantic_analysis.stats import SemanticCaptionStats


def prepare_clusters_for_agent(summary: dict[int, dict]) -> list[dict]:
    return [
        {"group_id": str(cluster_id), "captions": cluster["captions"]}
        for cluster_id, cluster in summary.items()
    ]


class SemanticCaptionInterpreter(BaseInterpreter[SemanticCaptionStats, ReportContent]):
    def interpret(self) -> AgentRunResult[Any]:
        if not self.agent:
            raise ValueError("Agent is not set for SemanticCaptionInterpreter")

        clusters = prepare_clusters_for_agent(self.stats.cluster_caption_summary)
        return self.agent.run_sync(json.dumps(clusters))

    def format(
        self, section: str, sub_section: str, agent_output: AgentRunResult[Any]
    ) -> ReportContent:
        if self.stats.clustered_df is None or self.stats.clustered_df.empty:
            raise ValueError("Clustered DataFrame is not set in stats")

        full_text = ""
        data = {}
        chart_idx = 1

        for group in agent_output.data:
            group_id = int(group.group_id)
            matching_rows = self.stats.clustered_df[
                self.stats.clustered_df["semantic_cluster"] == group_id
            ]

            asset_ids: list[str] = [
                str(row["asset_id"]) for _, row in matching_rows.iterrows()
            ]
            if not asset_ids:
                continue

            chart_id = f"chart-{chart_idx}"
            chart_idx += 1
            data[chart_id] = Assets(type="asset-list", ids=asset_ids)

            named_entities = (
                ", ".join(group.named_entities) if group.named_entities else "None"
            )

            full_text += clean_newlines(f"""### Semantic cluster **{group.group_name}**

{group.insights}
Named entities extracted from captions: {named_entities}
Associated images: {data_reference_tag(chart_id)}

""")

        return ReportContent(
            name="Semantic Caption Clustering",
            section=section,
            sub_section=sub_section,
            text=full_text.strip(),
            data=data,
            potential_actions=[PossibleActions.TAG],
        )
