from typing import Any

import httpx
import orjson

from config.settings import settings


def callback_platform(
    http_client: httpx.Client,
    url: str,
    data: dict[str, Any],
) -> httpx.Response:
    response = http_client.post(
        url=url,
        content=orjson.dumps(data),
        headers={
            "Authorization": f"Bearer {settings.api_key}",
            "Content-Type": "application/json",
        },
    )
    response.raise_for_status()
    return response
