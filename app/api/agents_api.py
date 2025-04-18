from datetime import UTC, datetime

from fastapi import APIRouter

from api.schemas import AnalysisRequest, ChatRequest
from services.context import get_context_service
from services.report_enums import SectionName, SubSectionName
from services.tasks import (
    compute_analysis_task,
    process_chat_message_task,
    process_report_chat_message_task,
)

router = APIRouter(prefix="")


@router.post("/compute-analysis")
def compute_analysis(request: AnalysisRequest):
    context_service = get_context_service(
        request.api_token,
        dataset_version_id=request.dataset_version_id,
        callback_url=request.callback_url,
        organization_id=request.organization_id,
        report_object_name=request.report_object_name,
        chat_messages_object_name=request.chat_messages_object_name,
        report_id=request.report_id,
    )
    report = {
        "name": "Report",
        "computing": True,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "sections": [
            {
                "name": SectionName.DATASET_OVERVIEW,
                "sub_sections": [
                    {"name": SubSectionName.OVERVIEW.name},
                ],
            },
            {
                "name": SectionName.IMAGE_QUALITY,
                "sub_sections": [
                    {"name": SubSectionName.IMAGE_STATISTICS_ANALYSIS.name},
                    {"name": SubSectionName.IMAGE_QUALITY_ISSUES.name},
                ],
            },
            {
                "name": SectionName.ANNOTATION_QUALITY,
                "sub_sections": [
                    {"name": SubSectionName.CLASS_ANALYSIS.name},
                    {"name": SubSectionName.SINGLE_OBJECT_ANALYSIS.name},
                    {"name": SubSectionName.ANNOTATION_OUTLIERS_ANALYSIS.name},
                    {"name": SubSectionName.INTER_CLASS_RELATION_ANALYSIS.name},
                ],
            },
        ],
        "content": [],
    }
    context_service.sync_context(report)
    context_service.upload_context_to_s3()
    compute_analysis_task.delay(
        api_token=request.api_token,
        callback_url=request.callback_url,
        organization_id=request.organization_id,
        dataset_version_id=request.dataset_version_id,
        report_id=request.report_id,
        report_object_name=request.report_object_name,
        chat_messages_object_name=request.chat_messages_object_name,
    )


@router.post("/chat")
def chat(request: ChatRequest):
    dataset_version_id = request.context["dataset_version_id"]
    if "report_id" in request.context:
        report_id = request.context["report_id"]
        report_object_name = request.context["report_object_name"]
        process_report_chat_message_task.delay(
            api_token=request.api_token,
            callback_url=request.callback_url,
            organization_id=request.organization_id,
            dataset_version_id=dataset_version_id,
            report_id=report_id,
            report_object_name=report_object_name,
            chat_messages_object_name=request.chat_messages_object_name,
            message=request.message,
        )
    else:
        process_chat_message_task.delay(
            api_token=request.api_token,
            callback_url=request.callback_url,
            organization_id=request.organization_id,
            dataset_version_id=dataset_version_id,
            chat_messages_object_name=request.chat_messages_object_name,
            message=request.message,
        )
