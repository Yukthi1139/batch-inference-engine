import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from fastapi import status

import app.inference_client as inference_client
from app.inference_client import PermanentInferenceError, RetryableInferenceError
import app.main as main


pytestmark = pytest.mark.asyncio(loop_scope="module")

SAMPLE_BATCH_PATH = "examples/sample_batch.json"
PROMPTS = [
    "Explain machine learning.",
    "What is batch inference?",
    "What is an API?",
]
TERMINAL_STATES = {"completed", "completed_with_errors", "failed"}


@pytest_asyncio.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture(loop_scope="module", autouse=True)
async def reset_job_state() -> AsyncIterator[None]:
    main.jobs.clear()
    main.results.clear()

    yield

    for _ in range(200):
        if all(job.status in TERMINAL_STATES for job in main.jobs.values()):
            break
        await asyncio.sleep(0.005)
    else:
        raise AssertionError("background jobs did not reach a terminal state")

    main.jobs.clear()
    main.results.clear()


@pytest.fixture(autouse=True)
def mock_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_inference(prompt: str) -> str:
        return f"mock response for: {prompt}"

    monkeypatch.setattr(main, "run_inference", fake_inference)


async def wait_for_terminal(client: httpx.AsyncClient, job_id: str) -> dict:
    for _ in range(200):
        response = await client.get(f"/job/{job_id}/status")
        payload = response.json()
        if payload["status"] in TERMINAL_STATES:
            return payload
        await asyncio.sleep(0.005)
    raise AssertionError(f"job {job_id} did not reach a terminal state")


async def submit_sample(client: httpx.AsyncClient) -> dict:
    response = await client.post("/jobs", json={"file_path": SAMPLE_BATCH_PATH})
    assert response.status_code == status.HTTP_201_CREATED
    return response.json()


async def test_inference_client_sends_official_request(monkeypatch: pytest.MonkeyPatch) -> None:
    request = {}

    class MockAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            assert timeout == inference_client.REQUEST_TIMEOUT_SECONDS

        async def __aenter__(self) -> "MockAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, endpoint: str, *, headers: dict, json: dict) -> httpx.Response:
            request.update(endpoint=endpoint, headers=headers, json=json)
            return httpx.Response(
                status_code=200,
                json={"choices": [{"message": {"content": "generated text"}}]},
            )

    monkeypatch.setenv("MODEL_ACCESS_KEY", "test-model-access-key")
    monkeypatch.setenv("INFERENCE_ENDPOINT", "https://example.test/inference")
    monkeypatch.setattr(inference_client.httpx, "AsyncClient", MockAsyncClient)

    result = await inference_client.run_inference("Summarize this")

    assert result == "generated text"
    assert request == {
        "endpoint": "https://example.test/inference",
        "headers": {
            "Authorization": "Bearer test-model-access-key",
            "Content-Type": "application/json",
        },
        "json": {
            "model": "openai-gpt-oss-20b",
            "messages": [{"role": "user", "content": "Summarize this"}],
            "max_completion_tokens": 300,
            "reasoning_effort": "low",
        },
    }


@pytest.mark.parametrize("missing", ["MODEL_ACCESS_KEY", "INFERENCE_ENDPOINT"])
async def test_inference_client_rejects_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setenv("MODEL_ACCESS_KEY", "test-model-access-key")
    monkeypatch.setenv("INFERENCE_ENDPOINT", "https://example.test/inference")
    monkeypatch.delenv(missing)

    with pytest.raises(PermanentInferenceError, match="is not configured"):
        await inference_client.run_inference("prompt")


@pytest.mark.parametrize("status_code", [403, 429, 500, 503])
async def test_inference_client_classifies_retryable_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    class MockAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            pass

        async def __aenter__(self) -> "MockAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                status_code=status_code,
                json={"error": {"message": "provider message"}},
            )

    monkeypatch.setenv("MODEL_ACCESS_KEY", "test-model-access-key")
    monkeypatch.setenv("INFERENCE_ENDPOINT", "https://example.test/inference")
    monkeypatch.setattr(inference_client.httpx, "AsyncClient", MockAsyncClient)

    with pytest.raises(
        RetryableInferenceError,
        match=rf"{status_code}.*provider message",
    ):
        await inference_client.run_inference("prompt")


@pytest.mark.parametrize("status_code", [400, 401])
async def test_inference_client_classifies_permanent_statuses(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int,
) -> None:
    class MockAsyncClient:
        def __init__(self, *, timeout: float) -> None:
            pass

        async def __aenter__(self) -> "MockAsyncClient":
            return self

        async def __aexit__(self, *args: object) -> None:
            return None

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return httpx.Response(
                status_code=status_code,
                json={"error": {"message": "provider message"}},
            )

    monkeypatch.setenv("MODEL_ACCESS_KEY", "test-model-access-key")
    monkeypatch.setenv("INFERENCE_ENDPOINT", "https://example.test/inference")
    monkeypatch.setattr(inference_client.httpx, "AsyncClient", MockAsyncClient)

    with pytest.raises(
        PermanentInferenceError,
        match=rf"{status_code}.*provider message",
    ):
        await inference_client.run_inference("prompt")


async def test_health_endpoint(client: httpx.AsyncClient) -> None:
    response = await client.get("/health")

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"status": "running"}


async def test_valid_batch_submission(client: httpx.AsyncClient) -> None:
    payload = await submit_sample(client)

    assert payload["job_id"]
    assert payload["status"] == "queued"
    assert payload["total_prompts"] == 3


@pytest.mark.parametrize(
    "content",
    ["{not valid json", "{}", "[]", '["", 42]'],
)
async def test_invalid_batch_files(
    client: httpx.AsyncClient,
    tmp_path: Path,
    content: str,
) -> None:
    batch_path = tmp_path / "invalid.json"
    batch_path.write_text(content, encoding="utf-8")

    response = await client.post("/jobs", json={"file_path": str(batch_path)})

    assert response.status_code == status.HTTP_400_BAD_REQUEST


async def test_invalid_file_path_and_extension(
    client: httpx.AsyncClient,
    tmp_path: Path,
) -> None:
    nonexistent = await client.post(
        "/jobs", json={"file_path": str(tmp_path / "missing.json")}
    )
    text_file = tmp_path / "batch.txt"
    text_file.write_text("[]", encoding="utf-8")
    wrong_extension = await client.post(
        "/jobs", json={"file_path": str(text_file)}
    )

    assert nonexistent.status_code == status.HTTP_400_BAD_REQUEST
    assert wrong_extension.status_code == status.HTTP_400_BAD_REQUEST


async def test_status_endpoint_reaches_completed(client: httpx.AsyncClient) -> None:
    payload = await submit_sample(client)

    status_response = await client.get(f"/job/{payload['job_id']}/status")
    completed = await wait_for_terminal(client, payload["job_id"])

    assert status_response.status_code == status.HTTP_200_OK
    assert completed["status"] == "completed"
    assert completed["completed_prompts"] == completed["total_prompts"] == 3
    assert completed["failed_prompts"] == 0


async def test_status_endpoint_rejects_unknown_job(client: httpx.AsyncClient) -> None:
    response = await client.get("/job/unknown-job/status")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_successful_results_preserve_order(client: httpx.AsyncClient) -> None:
    original = main.run_inference

    async def delayed_inference(prompt: str) -> str:
        delays = {
            PROMPTS[0]: 0.03,
            PROMPTS[1]: 0.01,
            PROMPTS[2]: 0.02,
        }
        await asyncio.sleep(delays[prompt])
        return await original(prompt)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(main, "run_inference", delayed_inference)
    try:
        payload = await submit_sample(client)
        completed = await wait_for_terminal(client, payload["job_id"])
    finally:
        monkeypatch.undo()

    assert completed["status"] == "completed"
    download = await client.get(f"/job/{payload['job_id']}/download")
    results = download.json()["results"]
    assert [item["prompt"] for item in results] == PROMPTS


async def test_download_rejects_incomplete_job(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocked_inference(prompt: str) -> str:
        started.set()
        await release.wait()
        return f"mock response for: {prompt}"

    monkeypatch.setattr(main, "run_inference", blocked_inference)
    payload = await submit_sample(client)
    await asyncio.wait_for(started.wait(), timeout=1)

    response = await client.get(f"/job/{payload['job_id']}/download")
    release.set()

    assert response.status_code == status.HTTP_409_CONFLICT
    assert "not complete yet" in response.json()["detail"]
    await wait_for_terminal(client, payload["job_id"])


async def test_completed_download_has_json_attachment(client: httpx.AsyncClient) -> None:
    payload = await submit_sample(client)
    await wait_for_terminal(client, payload["job_id"])

    response = await client.get(f"/job/{payload['job_id']}/download")
    downloaded = response.json()

    assert response.status_code == status.HTTP_200_OK
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["content-disposition"] == (
        f'attachment; filename="job_{payload["job_id"]}_results.json"'
    )
    assert downloaded["status"] == "completed"
    assert [item["prompt"] for item in downloaded["results"]] == PROMPTS


async def test_download_rejects_unknown_job(client: httpx.AsyncClient) -> None:
    response = await client.get("/job/unknown-job/download")

    assert response.status_code == status.HTTP_404_NOT_FOUND


async def test_transient_failure_is_retried(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = main.run_inference
    attempts = 0

    async def transient_inference(prompt: str) -> str:
        nonlocal attempts
        if prompt == PROMPTS[0]:
            attempts += 1
            if attempts < 2:
                raise RetryableInferenceError("temporary failure")
        return await original(prompt)

    monkeypatch.setattr(main, "run_inference", transient_inference)
    payload = await submit_sample(client)
    completed = await wait_for_terminal(client, payload["job_id"])

    assert attempts == 2
    assert completed["status"] == "completed"
    assert completed["failed_prompts"] == 0


async def test_partial_failure_isolated_and_ordered(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = main.run_inference

    async def partial_failure(prompt: str) -> str:
        if prompt == PROMPTS[1]:
            raise PermanentInferenceError("permanent prompt failure")
        return await original(prompt)

    monkeypatch.setattr(main, "run_inference", partial_failure)
    payload = await submit_sample(client)
    completed = await wait_for_terminal(client, payload["job_id"])

    assert completed["status"] == "completed_with_errors"
    assert completed["failed_prompts"] == 1
    assert completed["completed_prompts"] == 2
    results = main.results[payload["job_id"]]
    assert [item["prompt"] for item in results if item] == PROMPTS
    assert results[1] == {
        "prompt": PROMPTS[1],
        "response": None,
        "error": "permanent prompt failure",
    }


async def test_complete_failure_sets_failed_status(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def permanent_failure(prompt: str) -> str:
        raise PermanentInferenceError("permanent failure")

    monkeypatch.setattr(main, "run_inference", permanent_failure)
    payload = await submit_sample(client)
    completed = await wait_for_terminal(client, payload["job_id"])

    assert completed["status"] == "failed"
    assert completed["failed_prompts"] == completed["total_prompts"] == 3


async def test_global_concurrency_limit_across_jobs(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    peak = 0

    async def instrumented_inference(prompt: str) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await asyncio.sleep(0.02)
            return f"mock response for: {prompt}"
        finally:
            active -= 1

    monkeypatch.setattr(main, "run_inference", instrumented_inference)
    first, second = await asyncio.gather(
        submit_sample(client),
        submit_sample(client),
    )
    await asyncio.gather(
        wait_for_terminal(client, first["job_id"]),
        wait_for_terminal(client, second["job_id"]),
    )

    assert peak <= main.MAX_WORKERS


async def test_download_allows_completed_with_errors(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def permanent_failure(prompt: str) -> str:
        raise PermanentInferenceError("permanent failure")

    monkeypatch.setattr(main, "run_inference", permanent_failure)
    payload = await submit_sample(client)
    await wait_for_terminal(client, payload["job_id"])

    response = await client.get(f"/job/{payload['job_id']}/download")

    assert response.status_code == status.HTTP_200_OK
    assert json.loads(response.content)["status"] == "failed"