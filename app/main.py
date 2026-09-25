"""Application entry point for the batch inference engine."""

import asyncio
import json
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .inference_client import RetryableInferenceError, run_inference


class JobSubmissionRequest(BaseModel):
    file_path: str = Field(min_length=1)


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: str
    total_prompts: int


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total_prompts: int
    completed_prompts: int
    failed_prompts: int


class JobDownloadResponse(BaseModel):
    job_id: str
    status: str
    total_prompts: int
    results: list[dict[str, str | None]]


app = FastAPI(title="Batch Inference Engine")
MAX_WORKERS = 5
MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 0.01
processing_semaphore = asyncio.Semaphore(MAX_WORKERS)
jobs: dict[str, JobStatusResponse] = {}
results: dict[str, list[dict[str, str | None] | None]] = {}


async def run_inference_with_retries(prompt: str) -> str:
    for attempt in range(MAX_RETRIES + 1):
        try:
            async with processing_semaphore:
                return await run_inference(prompt)
        except RetryableInferenceError:
            if attempt == MAX_RETRIES:
                raise
            delay = BASE_BACKOFF_SECONDS * (2**attempt)
            await asyncio.sleep(delay)

    raise RuntimeError("inference retry loop ended unexpectedly")


async def worker(
    job_id: str,
    queue: asyncio.Queue[tuple[int, str]],
    job_results: list[dict[str, str | None] | None],
) -> None:
    while True:
        index, prompt = await queue.get()
        try:
            try:
                response = await run_inference_with_retries(prompt)
            except Exception as exc:
                error_message = str(exc) or exc.__class__.__name__
                job_results[index] = {
                    "prompt": prompt,
                    "response": None,
                    "error": error_message,
                }
                jobs[job_id].failed_prompts += 1
            else:
                job_results[index] = {
                    "prompt": prompt,
                    "response": response,
                    "error": None,
                }
                jobs[job_id].completed_prompts += 1
        finally:
            queue.task_done()


async def process_job(job_id: str, prompts: list[str]) -> None:
    jobs[job_id].status = "running"
    job_results: list[dict[str, str | None] | None] = [None] * len(prompts)
    results[job_id] = job_results
    queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()

    for index, prompt in enumerate(prompts):
        await queue.put((index, prompt))

    workers = [
        asyncio.create_task(worker(job_id, queue, job_results))
        for _ in range(MAX_WORKERS)
    ]
    await queue.join()

    for worker_task in workers:
        worker_task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    if jobs[job_id].failed_prompts == 0:
        jobs[job_id].status = "completed"
    elif jobs[job_id].completed_prompts > 0:
        jobs[job_id].status = "completed_with_errors"
    else:
        jobs[job_id].status = "failed"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "running"}


@app.post(
    "/jobs",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def submit_job(request: JobSubmissionRequest) -> JobSubmissionResponse:
    file_path = Path(request.file_path)
    if file_path.suffix.lower() != ".json":
        raise HTTPException(status_code=400, detail="file_path must reference a JSON file")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="file_path does not exist")

    try:
        with file_path.open(encoding="utf-8") as file:
            prompts = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="file_path is not valid JSON") from exc

    if not isinstance(prompts, list) or not prompts:
        raise HTTPException(status_code=400, detail="JSON file must contain a non-empty array")
    if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise HTTPException(
            status_code=400,
            detail="JSON array must contain only non-empty strings",
        )

    job_id = str(uuid4())
    jobs[job_id] = JobStatusResponse(
        job_id=job_id,
        status="queued",
        total_prompts=len(prompts),
        completed_prompts=0,
        failed_prompts=0,
    )
    asyncio.create_task(process_job(job_id, prompts))

    return JobSubmissionResponse(
        job_id=job_id,
        status="queued",
        total_prompts=len(prompts),
    )


@app.get("/job/{job_id}/status", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found")
    return jobs[job_id]


@app.get("/job/{job_id}/download")
def download_job_results(job_id: str) -> JSONResponse:
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' was not found")
    if jobs[job_id].status not in {"completed", "completed_with_errors", "failed"}:
        raise HTTPException(
            status_code=409,
            detail=f"Job '{job_id}' is not complete yet",
        )

    response = JobDownloadResponse(
        job_id=job_id,
        status=jobs[job_id].status,
        total_prompts=jobs[job_id].total_prompts,
        results=[result for result in results[job_id] if result is not None],
    )
    return JSONResponse(
        content=response.model_dump(),
        headers={
            "Content-Disposition": (
                f'attachment; filename="job_{job_id}_results.json"'
            )
        },
    )