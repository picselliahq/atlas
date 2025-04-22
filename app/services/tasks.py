import logging
from pathlib import Path

import pandas as pd
import yaml
from celery import shared_task
from celery.utils.log import get_task_logger
from picsellia.exceptions import NoDataError
from smolagents import PromptTemplates, ToolCallingAgent

from agents.ai_models import chat_model
from agents.common.models.contents import ReportEvent
from agents.common.tool.fork_dataset_version import CreateNewDatasetVersion
from agents.common.tool.get_dataset_analysis_result import GetDatasetAnalysisReport
from agents.mcp_chat_agent.agent import create_mcp_agent
from agents.mcp_chat_agent.tools.final_answer import FinalAnswerTool
from services.chat import build_message
from services.context import ContextService, get_context_service

logger: logging.Logger = get_task_logger(__name__)


def report_agent(context_service: ContextService, message: str):
    chat_session = context_service.get_chat_session()
    human_message = build_message("human", message)
    context_service.add_message_to_chat_session(human_message)
    dataset_report_tool = GetDatasetAnalysisReport(context=context_service)
    create_new_dataset_version = CreateNewDatasetVersion(context=context_service)

    yaml_path = Path(__file__).parent / "system_prompt" / "tool_calling_prompt.yaml"
    with open(yaml_path) as f:
        tool_calling_prompt = yaml.safe_load(f)

    tool_calling_prompt = PromptTemplates(**tool_calling_prompt)

    agent = ToolCallingAgent(
        name="ReportAgent",
        tools=[dataset_report_tool, create_new_dataset_version],
        model=chat_model,
        max_steps=5,
        prompt_templates=tool_calling_prompt,
    )
    agent.tools.setdefault("final_answer", FinalAnswerTool())
    result = agent.run(
        message,
        additional_args={
            "chat_session": chat_session,
        },
    )
    machine_response = build_message("machine", result)
    context_service.add_message_to_chat_session(machine_response)
    context_service.send_chat_message_to_platform(machine_response)
    context_service.upload_chat_session_to_s3()


def mcp_agent(context_service: ContextService, message: str):
    human_message = build_message("human", message)
    context_service.add_message_to_chat_session(human_message)
    agent = create_mcp_agent(context_service=context_service)
    agent.tools.setdefault("final_answer", FinalAnswerTool())
    result = agent.run(message)
    machine_response = build_message("machine", result)
    context_service.add_message_to_chat_session(machine_response)
    context_service.send_chat_message_to_platform(machine_response)
    context_service.upload_chat_session_to_s3()


@shared_task(name="process_report_chat_message_task")
def process_report_chat_message_task(
    api_token: str,
    callback_url: str,
    organization_id: str,
    dataset_version_id: str,
    report_id: str,
    report_object_name: str,
    chat_messages_object_name: str,
    message: str,
):
    context_service = get_context_service(
        api_token,
        dataset_version_id=dataset_version_id,
        callback_url=callback_url,
        organization_id=organization_id,
        report_object_name=report_object_name,
        chat_messages_object_name=chat_messages_object_name,
        report_id=report_id,
    )
    context = context_service.get_context_dict()
    if not context:
        mcp_agent(context_service=context_service, message=message)
    else:
        report_agent(context_service=context_service, message=message)


@shared_task(name="process_chat_message_task")
def process_chat_message_task(
    api_token: str,
    callback_url: str,
    organization_id: str,
    dataset_version_id: str,
    chat_messages_object_name: str,
    message: str,
):
    context_service = get_context_service(
        api_token,
        dataset_version_id=dataset_version_id,
        callback_url=callback_url,
        organization_id=organization_id,
        report_id=None,
        report_object_name=None,
        chat_messages_object_name=chat_messages_object_name,
    )
    mcp_agent(context_service=context_service, message=message)


@shared_task(name="compute_analysis_task")
def compute_analysis_task(
    api_token: str,
    callback_url: str,
    organization_id: str,
    dataset_version_id: str,
    report_id: str,
    report_object_name: str,
    chat_messages_object_name: str,
):
    from agents.annotation_agent.main import AnnotationAnalysisTool
    from agents.image_agent.main import ImageAnalyticsTool

    context = get_context_service(
        api_token,
        dataset_version_id=dataset_version_id,
        callback_url=callback_url,
        organization_id=organization_id,
        report_object_name=report_object_name,
        chat_messages_object_name=chat_messages_object_name,
        report_id=report_id,
    )
    context.get_context_dict()
    image_metadata: pd.DataFrame = ImageAnalyticsTool(context_service=context).forward()
    context.upload_context_to_s3()
    dataset_version = context.client.get_dataset_version_by_id(dataset_version_id)
    try:
        annotations = dataset_version.list_annotations(limit=1)
    except NoDataError:
        annotations = []

    if len(annotations) > 0:
        print(f"🖼️ Found {len(annotations)} annotations in the dataset version.")
        AnnotationAnalysisTool(context_service=context).forward(
            image_metadata=image_metadata
        )

    event = ReportEvent(event="computation_done", message="Analysis done.")
    context.sync_content(event)
    context.upload_context_to_s3()
