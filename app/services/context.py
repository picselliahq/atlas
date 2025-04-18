import json
import logging
import os
import tempfile
from collections import defaultdict
from typing import Literal
from urllib.parse import urlparse
from uuid import UUID

import pandas as pd
from picsellia import Client
from picsellia.types.enums import ObjectDataType

from agents.common.models.contents import (
    ChatMessage,
    ReportContent,
    ReportError,
    ReportEvent,
)
from api.clients.hinokuni import callback_platform
from api.clients.httpx_client import client as httpx_client
from api.clients.picsellia_sdk import get_client
from services.file_handler import FileService
from services.redis_config import redis_client as cache

logger = logging.getLogger(__name__)


def get_protocol_and_domain(url: str) -> str:
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}"


class ContextService:
    def __init__(
        self,
        client: Client,
        callback_url: str,
        dataset_id: str,
        report_id: str | None,
        context_object_name: str | None,
        chat_session_object_name: str,
    ):
        self.client = client
        self.dataset_id = dataset_id
        self.report_id = report_id
        self.callback_url = callback_url
        self.context: dict = defaultdict(dict)
        self.chat_session: dict = {}
        self.md_report = ""
        self.metadata = pd.DataFrame()
        self.file_service = FileService(client)
        if context_object_name:
            self.context_object_name = context_object_name
        self.chat_session_object_name = chat_session_object_name
        if report_id:
            self.annotation_agent_data_object_name = (
                client.connexion.generate_report_object_name(
                    filename="annotation_agent_data.csv",
                    object_name_type=ObjectDataType.AGENTS_REPORT,
                    dataset_version_id=UUID(dataset_id),
                    report_id=UUID(report_id),
                )
            )
            self.image_agent_data_object_name = (
                client.connexion.generate_report_object_name(
                    filename="image_agent_data.csv",
                    object_name_type=ObjectDataType.AGENTS_REPORT,
                    dataset_version_id=UUID(dataset_id),
                    report_id=UUID(report_id),
                )
            )

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

    def _upload_to_s3(
        self,
        data: dict | pd.DataFrame,
        object_name: str,
        cache_key: str,
        file_type: str = "json",
    ) -> None:
        """
        Upload a file to S3 and update the cache.
        Args:
            data: The data to upload (Dict for context, DataFrame for metadata)
            object_name: The object_name of the file to upload
            file_type: The type of file ("json" or "csv")
        """
        with tempfile.NamedTemporaryFile(
            suffix=f".{file_type}", delete=False
        ) as temp_file:
            temp_path = temp_file.name

        try:
            if file_type == "json":
                with open(temp_path, "w") as f:
                    json.dump(data, f)
                self.set_in_cache(data, cache_key)
            else:
                data.to_csv(temp_path, index=False)  # type: ignore[union-attr]
                self.set_in_cache(data, cache_key)

            self.file_service.upload(temp_path, object_name)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def upload_metadata_to_s3(self, agent_type: str) -> None:
        """
        Upload the metadata DataFrame to S3 as a CSV file.
        """
        self._upload_to_s3(
            self.metadata,
            self._get_metadata_object_name(agent_type),
            self._get_metadata_cache_key(),
            "csv",
        )

    def upload_context_to_s3(self) -> None:
        """
        Upload the context dictionary to S3 as a JSON file.
        """
        self._upload_to_s3(
            self.context,
            self.context_object_name,
            self._get_context_cache_key(),
            "json",
        )

    def upload_chat_session_to_s3(self) -> None:
        """
        Upload the chat session dictionary to S3 as a JSON file.
        """
        self._upload_to_s3(
            self.chat_session,
            self.chat_session_object_name,
            self._get_chat_session_cache_key(),
            "json",
        )

    def download_metadata_from_s3(self, agent_type: str) -> pd.DataFrame:
        """
        Download the metadata CSV file from S3 and cache it.
        Returns:
            pd.DataFrame: The downloaded metadata
        """
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            object_name = self._get_metadata_object_name(agent_type)
            self.file_service.download(temp_path, object_name)

            self.metadata = pd.read_csv(temp_path)
            self.set_in_cache(self.metadata, self._get_metadata_cache_key())
        except Exception as e:
            logger.warning(f"Failed to download metadata from S3: {e}")
            self.metadata = {}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return self.metadata

    def _get_metadata_object_name(self, agent_type: str) -> str:
        if agent_type == "annotation_agent":
            return self.annotation_agent_data_object_name
        elif agent_type == "image_agent":
            return self.image_agent_data_object_name
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    def download_context_from_s3(self) -> dict:
        """
        Download the context JSON file from S3 and cache it.
        Returns:
            Dict: The downloaded context
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            object_name = self.context_object_name
            self.file_service.download(temp_path, object_name)

            with open(temp_path) as f:
                self.context = json.load(f)
                self.set_in_cache(self.context, self._get_context_cache_key())
        except Exception as e:
            logger.warning(f"Failed to download context from S3: {e}")
            self.context = {}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return self.context

    def download_chat_session_from_s3(self) -> dict:
        """
        Download the chat session JSON file from S3 and cache it.
        Returns:
            Dict: The downloaded chat session
        """
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            object_name = self.chat_session_object_name
            self.file_service.download(temp_path, object_name)

            with open(temp_path) as f:
                self.chat_session = json.load(f)
                self.set_in_cache(self.chat_session, self._get_chat_session_cache_key())
        except Exception as e:
            logger.warning(f"Failed to download chat from S3: {e}")
            self.chat_session = {}
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return self.chat_session

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

        return self.download_context_from_s3()

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
        return self.download_chat_session_from_s3()

    def get_metadata(self, agent_type: str) -> pd.DataFrame:
        """
        Get the metadata by trying in order:
        1. Return if metadata is not empty
        2. Try to get from cache
        3. Download from S3 if all else fails
        """

        # if not self.metadata.empty:
        #     return self.metadata
        #
        # cached_metadata = self._get_from_cache(self._get_metadata_cache_key())
        # if cached_metadata is not None:
        #     self.metadata = cached_metadata
        #     return self.metadata

        return self.download_metadata_from_s3(agent_type=agent_type)

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

    def sync_metadata(self, df: pd.DataFrame) -> None:
        """
        Sync the metadata DataFrame with content.
        Args:
            df: The metadata DataFrame to sync
        """
        self.metadata = df

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
            data = self._sync_report_content(content)
            data_type = "content"
        elif isinstance(content, ReportError):
            data = self._sync_error_content(content)
            data_type = "error"
        elif isinstance(content, ReportEvent):
            data = self._sync_event_content(content)
            data_type = "event"
        else:
            raise ValueError("Invalid content type")
        self._callback_platform(data_type, data)  # type: ignore[arg-type]

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

    def _callback_platform(
        self, type: Literal["content", "error", "event"], data: dict
    ):
        data = {"type": type, **data}
        callback_platform(
            http_client=httpx_client,
            url=self.callback_url,
            data=data,
        )

    def add_message_to_chat_session(self, message: ChatMessage) -> None:
        if "chat_messages" not in self.chat_session:
            self.chat_session["chat_messages"] = []
        self.chat_session["chat_messages"].append(message.model_dump())

    def send_chat_message_to_platform(self, message: ChatMessage) -> None:
        callback_platform(
            http_client=httpx_client,
            url=self.callback_url,
            data=message.model_dump(),
        )

    def _build_md_report(self) -> str:
        formatted_report: dict = {}
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


def get_context_service(
    api_token: str,
    callback_url: str,
    organization_id: str,
    dataset_version_id: str,
    report_id: str | None,
    report_object_name: str | None,
    chat_messages_object_name: str,
):
    dataset_id = dataset_version_id
    host = get_protocol_and_domain(callback_url)
    client = get_client(host, api_token, organization_id)
    context = ContextService(
        client=client,
        callback_url=callback_url,
        dataset_id=dataset_id,
        report_id=report_id,
        context_object_name=report_object_name,
        chat_session_object_name=chat_messages_object_name,
    )
    return context
