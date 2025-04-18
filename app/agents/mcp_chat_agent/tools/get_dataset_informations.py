import textwrap
from datetime import datetime

from agents.common.tool.mcp_tool import BaseMCPTool
from services.context import ContextService


class GetMetadatasetInformation(BaseMCPTool):
    name = "get_metadataset_informations"
    description = """
    Retrieve all the information about the metadataset content you need about the metadataset in order to answer the user query.
    Do not look for other content informations in another tool.
    """
    inputs = {
        "mode": {
            "type": "string",
            "description": "use mode='full' to get all the informations.",
            "nullable": "false",
        },
    }
    output_type = "string"

    def __init__(self, context: ContextService):
        super().__init__(context)
        self.metadata: dict[str, str | dict] = {}

    def _generate_report(self, with_campaign):
        created_at = datetime.fromisoformat(self.metadata["created_at"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        updated_at = datetime.fromisoformat(self.metadata["updated_at"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        report_lines = [
            "=== metadataset Report ===",
            f"ID: {self.metadata.get('id')}",
            f"Dataset Name {self.metadata.get('origin_name')}/{self.metadata.get('version')}",
            f"Type: {self.metadata.get('type')}",
            f"Forked from dataset: {self.metadata.get('origin_name')} (ID: {self.metadata.get('origin_id')})",
            f"Created at: {created_at}",
            f"Updated at: {updated_at}",
            f"Created by: {self.metadata['created_by'].get('username')}",
            f"Locked: {'Yes' if self.metadata.get('is_locked') else 'No'}",
            f"Number of Images in Dataset: {self.metadata.get('size')} samples",
            f"Can I use Visual Search on this dataset?: {'Yes' if self.metadata.get('visual_search_activated') else 'No'}",
            "",
            f"Number of Images Annotated: {self.metadata.get('nb_annotations', 'N/A')}",
            f"Number of instances annotated in total : {self.metadata.get('nb_objects', 'N/A')}",
            "",
            "Label Repartition:",
        ]
        label_repartition = self.metadata.get("label_repartition", {})
        for label, count in label_repartition.items():
            report_lines.append(f"  - {label}: {count}")
        report_lines.append("")
        report_lines.append(f"Labels ({len(self.metadata.get('labels', []))}):")
        for label in self.metadata.get("labels", []):
            label_line = f"  - {label.get('name')} (ID: {label.get('id')})"
            report_lines.append(label_line)

        if with_campaign:
            report_lines.append("Dataset is in an annotation Campaign.")
        else:
            report_lines.append("Dataset isn't in an annotation Campaign.")

        return textwrap.dedent("\n".join(report_lines))

    def forward(self, mode: str = "full") -> str:
        self.metadata = self.dataset_version.sync()
        self.metadata.update(self.dataset_version.retrieve_stats().dict())
        try:
            self.metadata.update(self.dataset_version.get_campaign().sync())
        except Exception:
            return self._generate_report(with_campaign=False)
        verbose_report = self._generate_report(with_campaign=True)
        print(verbose_report)
        return verbose_report
