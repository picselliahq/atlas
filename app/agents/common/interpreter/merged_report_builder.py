import re

from agents.common.models.contents import ReportContent


class MergedReportBuilder:
    def __init__(
        self,
        contents: list[ReportContent],
        section: str | None = None,
        sub_section: str | None = None,
        content_name: str | None = None,
    ):
        self.contents = contents
        self.content_name = content_name or contents[0].name
        self.section = section or contents[0].section
        self.sub_section = sub_section or contents[0].sub_section

    def build(self) -> ReportContent:
        full_text = ""
        full_data = {}
        chart_counter = 1

        for content in self.contents:
            part_text = content.text.strip()
            renamed_data = {}
            id_mapping = {}

            for original_id in content.data.keys():
                new_id = f"chart-{chart_counter}"
                id_mapping[original_id] = new_id
                chart_counter += 1

            def replace_chart_id(match, mapping=id_mapping):
                old_id = match.group(1)
                return f'<Data id="{mapping.get(old_id, old_id)}"/>'

            part_text = re.sub(
                r'<Data id="(chart-\d+)"/>',
                lambda match: replace_chart_id(match),
                part_text,
            )

            for original_id, chart in content.data.items():
                new_id = id_mapping[original_id]
                renamed_data[new_id] = chart

            full_text += f"\n{part_text}\n"
            full_data.update(renamed_data)

        return ReportContent(
            text=full_text.strip(),
            data=full_data,
            section=self.section,
            sub_section=self.sub_section,
            name=self.content_name,
            potential_actions=None,
        )


def subsection_title(name: str) -> str:
    return f"### {name}\n"
