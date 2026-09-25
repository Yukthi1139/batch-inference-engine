# Batch Inference Engine

An asynchronous FastAPI service for submitting a local JSON batch of prompts, processing them through DigitalOcean Serverless Inference, and downloading ordered results.

## Features

- Background job execution with immediate job IDs
- Global `MAX_WORKERS` concurrency limit
- Per-prompt retry with exponential backoff
- Isolated prompt failures and ordered results
- Job status tracking and JSON downloads
- Real asynchronous HTTP inference through `httpx`

## Architecture

```text
Client
  |
  +--> POST /jobs --> validate local JSON --> create queued job --> return job ID
                                      |
                                      v
                         background processor and prompt queue
                                      |
                                      v
                  global MAX_WORKERS semaphore --> inference client
                                      |                    |
                                      |                    v
                                      |          DigitalOcean Serverless API
                                      |                    |
                                      v                    v
                         retry/backoff --> ordered results --> terminal job state
                                      |
                    +-----------------+------------------+
                    v                                    v
             GET /job/{id}/status              GET /job/{id}/download
```

## Inference Configuration

The client uses the DigitalOcean Serverless Inference API with model `openai-gpt-oss-20b`.

Set these environment variables before starting the service:

```dotenv
MODEL_ACCESS_KEY=<your model access key>
INFERENCE_ENDPOINT=https://inference.do-ai.run/v1/chat/completions
```

No credentials are stored in source code, fixtures, or the repository. Use `.env.example` as the configuration reference; do not commit a real `.env` file.

The request uses `max_completion_tokens: 300` and `reasoning_effort: "low"`. The generated response is read from `choices[0].message.content`.

## Input and Results

The application accepts a top-level JSON array of non-empty prompt strings. The 1000-prompt example batch is:

```text
examples/input_batch.json
```

Submit a batch after starting the service:

```powershell
curl -X POST http://127.0.0.1:8000/jobs `
  -H "Content-Type: application/json" `
  -d '{"file_path":"examples/input_batch.json"}'
```

Check the returned job with `GET /job/{job_id}/status`, then download terminal results with `GET /job/{job_id}/download`.

Generated batch results are stored under `results/`, including:

- `results/input_batch_results.json`
- `results/retry_failed_batch_results.json`

## Validation Results

An initial live 1000-prompt run produced 969 successful results and 31 transient HTTP 403 failures. Based on the observed provider behavior, HTTP 403 was added to the retryable inference failures alongside HTTP 429, HTTP 5xx, and timeout/network errors. HTTP 400, HTTP 401, and other permanent client errors remain non-retryable.

Retrying the failed 31 prompts resulted in 31/31 successes.

## API

### `GET /health`

Returns the service health status.

### `POST /jobs`

Accepts a JSON body containing a local `file_path`, validates the batch, creates a background job, and returns HTTP `201` with a job ID.

### `GET /job/{job_id}/status`

Returns job state and completed/failed prompt counts.

### `GET /job/{job_id}/download`

Returns ordered prompt results for a terminal job as a JSON attachment.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API documentation is available at <http://127.0.0.1:8000/docs>.

Run the test suite with:

```powershell
python -m pytest -q
```
