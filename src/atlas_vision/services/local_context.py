import json
import logging
import uuid
from collections import defaultdict

import pandas as pd

from atlas_vision.agents.common.models.contents import (
    ChatMessage,
    ReportContent,
    ReportError,
    ReportEvent,
)
from atlas_vision.services.data_parser import DataParser
from atlas_vision.services.redis_config import redis_client as cache

logger = logging.getLogger(__name__)


class LocalContextService:
    def __init__(
        self,
        dataset_id: str,
        report_id: str | None,
    ):
        self.dataset_id = dataset_id
        if report_id:
            self.report_id = report_id
        else:
            self.report_id = uuid.uuid4()
        self.context: dict = defaultdict(dict)
        self.chat_session: dict = {}
        self.md_report = ""
        self.labels = {}
        self.image_metadata = pd.DataFrame()
        self.annotation_metadata = pd.DataFrame()
        self.data_parser = DataParser(self.report_id)

    def _get_report_cache_key(self) -> str:
        return f"dataset_report_{self.dataset_id}_{self.report_id}"

    def _get_context_cache_key(self) -> str:
        return f"dataset_context_{self.dataset_id}_{self.report_id}"

    def _get_chat_session_cache_key(self) -> str:
        return f"dataset_chat_session_{self.dataset_id}_{self.report_id}"

    def _get_metadata_cache_key(self) -> str:
        return f"dataset_metadata_{self.dataset_id}_{self.report_id}"

    def _get_from_cache(self, cache_key: str) -> dict | pd.DataFrame | None:
        cached_data = cache.get(cache_key)
        if cached_data is None:
            return None
        cached_data = cached_data.decode("utf-8")
        if cached_data:
            if cache_key == self._get_context_cache_key():
                return json.loads(cached_data)
            elif cache_key == self._get_chat_session_cache_key():
                return json.loads(cached_data)
            else:
                return pd.read_json(cached_data)
        return None

    def set_in_cache(self, data: dict | pd.DataFrame, cache_key: str) -> None:
        if isinstance(data, pd.DataFrame):
            cache.set(cache_key, data.to_json())
        else:
            cache.set(cache_key, json.dumps(data))

    def get_context_dict(self) -> dict:
        """
        Get the context by trying in order:
        1. Return if context is not empty
        2. Try to get from cache
        3. Download from S3 if all else fails
        """
        if self.context:
            return self.context

        cached_context = self._get_from_cache(self._get_context_cache_key())
        if cached_context:
            self.context = cached_context
            return self.context

    def get_chat_session(self) -> dict:
        """
        Get the chat session by trying in order:
        1. Return if chat session is not empty
        2. Try to get from cache
        3. Download from S3 if all else fails
        """
        if self.chat_session:
            return self.chat_session

        cached_chat_session = self._get_from_cache(self._get_chat_session_cache_key())
        if cached_chat_session:
            self.chat_session = cached_chat_session
            return self.chat_session

    def get_md_report(self):
        """
        Get the markdown report by trying in order:
        1. Return if metadata is not empty
        2. Try to get from cache

        It will be recomputer from the context if not stored in cache
        """
        if self.md_report:
            return self.md_report

        cached_md_report = self._get_from_cache(self._get_report_cache_key())
        if cached_md_report:
            self.md_report = cached_md_report
            return self.md_report
        return self._build_md_report()

    def sync_image_metadata(self, df: pd.DataFrame) -> None:
        """
        Sync the image metadata DataFrame with content.
        Args:
            df: The image metadata DataFrame to sync
        """
        self.image_metadata = df

    def sync_annotation_metadata(self, df: pd.DataFrame | None) -> None:
        """
        Sync the annotation metadata DataFrame with content.
        Args:
            df: The annotationmetadata DataFrame to sync
        """
        self.annotation_metadata = df

    def sync_labels(self, labels: dict) -> None:
        """
        Sync the labels dictionary with content.
        Args:
            labels: The labels dictionary to sync
        """
        self.labels = labels

    def sync_context(self, context: dict) -> None:
        """
        Sync the context dictionary with content.
        Args:
            context: The context dictionary to sync
        """
        self.context = context

    def sync_content(self, content: ReportContent | ReportError | ReportEvent):
        """
        Sync the content with the context.
        Args:
            content: The content to sync
        """
        if isinstance(content, ReportContent):
            self._sync_report_content(content)
        elif isinstance(content, ReportError):
            self._sync_error_content(content)
        elif isinstance(content, ReportEvent):
            self._sync_event_content(content)
        else:
            raise ValueError("Invalid content type")

    def _sync_report_content(self, content: ReportContent):
        if "content" not in self.context:
            self.context["content"] = []

        data = content.model_dump()
        self.context["content"].append(data)
        return data

    def _sync_error_content(self, error: ReportError):
        if "errors" not in self.context:
            self.context["errors"] = []

        data = error.model_dump()
        self.context["errors"].append(error.model_dump())
        return data

    def _sync_event_content(self, event: ReportEvent):
        self.context["computing"] = False
        return event.model_dump()

    def add_message_to_chat_session(self, message: ChatMessage) -> None:
        if "chat_messages" not in self.chat_session:
            self.chat_session["chat_messages"] = []
        self.chat_session["chat_messages"].append(message.model_dump())

    def _build_md_report(self) -> str:
        formatted_report: dict = {}
        chart_id = 0

        for section in self.context["sections"]:
            formatted_report[section["name"]] = {}
            for sub_section in section["sub_sections"]:
                formatted_report[section["name"]][sub_section["name"]] = ""
        for content in self.context["content"]:
            section = content["section"]
            sub_section = content["sub_section"]
            name = content["name"]

            content_md = formatted_report[section][sub_section]
            content_md += "\n"
            if name:
                content_md += f"###{name}\n"
            if content["text"]:
                content_md += f"{content['text']}\n"

            for name, value in content["data"].items():
                if value["type"] == "hist":
                    content_md += self.data_parser.parse_hist_chart(
                        value, name, chart_id
                    )
                elif value["type"] == "shape-list":
                    content_md += self.data_parser.parse_shape_list_data(
                        value["ids"], self.annotation_metadata, self.image_metadata
                    )
                chart_id += 1
            formatted_report[section][sub_section] = content_md
        report_md_string = ""
        for section in self.context["sections"]:
            report_md_string += "# " + section["name"] + "\n"
            for sub_section in section["sub_sections"]:
                report_md_string += "# " + sub_section["name"] + "\n"
                content = formatted_report[section["name"]][sub_section["name"]]
                report_md_string += content + "\n"
        self.md_report = report_md_string
        return self.md_report

    @staticmethod
    def _filter_image_dataframe_columns(metadata: pd.DataFrame):
        desired_columns = [
            "filename",
            "width",
            "height",
            "tags",
            "caption",
            "is_blurry",
            "is_corrupted",
            "blur_score",
            "file_size_bytes",
            "color",
            "luminance",
            "contrast",
        ]
        filtered_df = metadata[desired_columns]
        return filtered_df

    @staticmethod
    def _filter_annotation_dataframe_columns(metadata: pd.DataFrame):
        desired_columns = [
            "filename",
            "image_width",
            "image_height",
            "label",
            "x",
            "y",
            "w",
            "h",
            "x_norm",
            "y_norm",
        ]
        filtered_df = metadata[desired_columns]
        return filtered_df


def get_local_context_service(
    dataset_version_id: str,
    report_id: str | None = None,
):
    dataset_id = dataset_version_id
    context = LocalContextService(
        dataset_id=dataset_id,
        report_id=report_id,
    )
    return context
