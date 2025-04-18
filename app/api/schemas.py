from pydantic import BaseModel


class AnalysisRequest(BaseModel):
    api_token: str
    callback_url: str
    organization_id: str
    dataset_version_id: str
    report_id: str
    report_object_name: str
    chat_messages_object_name: str


class ChatRequest(BaseModel):
    api_token: str
    callback_url: str
    organization_id: str
    chat_id: str
    context: dict
    chat_messages_object_name: str
    message: str
