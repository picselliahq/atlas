from datetime import UTC, datetime

from agents.common.models.contents import ChatMessage


def build_message(sender: str, text: str, data: dict | None = None):
    return ChatMessage.model_validate(
        {
            "sender": sender,
            "text": text,
            "data": data,
            "created_at": datetime.now(tz=UTC).isoformat(),
        }
    )
