"""Asynchronous client for the DigitalOcean Serverless Inference API."""

import os
import re

import httpx

MODEL = "openai-gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 30.0


class InferenceError(Exception):
    """Base class for inference failures."""


class RetryableInferenceError(InferenceError):
    """An inference failure that may succeed when retried."""


class PermanentInferenceError(InferenceError):
    """An inference failure that should not be retried."""


def _short_error_detail(response: httpx.Response, model_access_key: str) -> str:
    try:
        body = response.json()
    except ValueError:
        body = response.text

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            detail = error.get("message") or error.get("detail")
        elif isinstance(error, str):
            detail = error
        else:
            detail = body.get("message") or body.get("detail")
    else:
        detail = body

    if not isinstance(detail, str):
        detail = str(detail) if detail is not None else "no response body"
    detail = " ".join(detail.split())
    detail = detail.replace(model_access_key, "[REDACTED]")
    detail = re.sub(
        r"(?i)(bearer\s+)\S+",
        r"\1[REDACTED]",
        detail,
    )
    return detail[:200] or "no response body"


async def run_inference(prompt: str) -> str:
    """Send one prompt to the configured inference endpoint."""
    model_access_key = os.getenv("MODEL_ACCESS_KEY")
    endpoint = os.getenv("INFERENCE_ENDPOINT")
    if not model_access_key:
        raise PermanentInferenceError("MODEL_ACCESS_KEY is not configured")
    if not endpoint:
        raise PermanentInferenceError("INFERENCE_ENDPOINT is not configured")

    request_json = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": 300,
        "reasoning_effort": "low",
    }
    headers = {
        "Authorization": f"Bearer {model_access_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(endpoint, headers=headers, json=request_json)
    except httpx.TimeoutException as exc:
        raise RetryableInferenceError("inference request timed out") from exc
    except httpx.RequestError as exc:
        raise RetryableInferenceError("inference request failed") from exc

    error_detail = _short_error_detail(response, model_access_key)
    if response.status_code in {403, 429} or response.status_code >= 500:
        raise RetryableInferenceError(
            f"inference service returned HTTP {response.status_code}: {error_detail}"
        )
    if response.status_code >= 400:
        raise PermanentInferenceError(
            f"inference request rejected with HTTP {response.status_code}: {error_detail}"
        )

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PermanentInferenceError("inference response had an invalid format") from exc
    if not isinstance(content, str):
        raise PermanentInferenceError("inference response content was not text")
    return content
