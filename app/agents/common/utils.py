import ast
from typing import Any

from agents.annotation_agent.common.get_context import PContext
from agents.common.models.contents import ReportContent


def convert_to_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    try:
        return ast.literal_eval(value)
    except Exception:
        return []


def data_reference_tag(chart_id: str) -> str:
    return f'<Data id="{chart_id}"/>'


def clean_newlines(text: str) -> str:
    cleaned_text = text.replace("\n\n", "\n")
    return cleaned_text


def dict_to_markdown_table(data: dict[str, str | int | float]) -> str:
    if not data:
        return ""

    # Headers and values
    headers = list(data.keys())
    values = [str(v) for v in data.values()]

    # Calculate max width for each column
    col_widths = [
        max(len(str(h)), len(str(v))) for h, v in zip(headers, values, strict=False)
    ]

    def format_row(row: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                f"{cell:<{w}}" for cell, w in zip(row, col_widths, strict=False)
            )
            + " |"
        )

    # Build the table
    header_row = format_row(headers)
    separator_row = "| " + " | ".join("-" * w for w in col_widths) + " |"
    value_row = format_row(values)

    return "\n".join([header_row, separator_row, value_row])


def sync_reports(pctx: PContext, reports: list[ReportContent]) -> None:
    for report in reports:
        pctx.context_service.sync_content(report)
