"""
RunPod Serverless handler for GLM-OCR via vLLM.

Starts vLLM HTTP server in the background, waits for it to be ready,
then forwards incoming RunPod jobs to the local OpenAI-compatible API.
"""

import os
import time
import subprocess
import threading
import logging
import requests
import runpod

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("handler")

VLLM_PORT = 8080
VLLM_URL = f"http://localhost:{VLLM_PORT}"
MODEL_NAME = os.getenv("MODEL_NAME", "zai-org/GLM-OCR")
MAX_MODEL_LEN = os.getenv("MAX_MODEL_LEN", "8192")
ENFORCE_EAGER = os.getenv("ENFORCE_EAGER", "1").lower() in {"1", "true", "yes"}


def stream_output(pipe):
    """Stream vLLM logs into worker logs for easier debugging."""
    try:
        for line in pipe:
            line = line.strip()
            if line:
                log.info("[vllm] %s", line)
    except Exception as exc:
        log.exception("Error while streaming vLLM logs: %s", exc)
    finally:
        pipe.close()


def start_vllm():
    """Start vLLM as a background process with log forwarding."""
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--allowed-local-media-path", "/",
        "--port", str(VLLM_PORT),
        "--max-model-len", MAX_MODEL_LEN,
    ]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")

    log.info("Starting vLLM: %s", " ".join(cmd))
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        t = threading.Thread(target=stream_output, args=(process.stdout,), daemon=True)
        t.start()

    return process


def wait_for_vllm(timeout=600):
    """Wait for vLLM to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{VLLM_URL}/health", timeout=2)
            if r.status_code == 200:
                log.info("vLLM is ready")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    raise TimeoutError(f"vLLM did not start within {timeout}s")


def handler(job):
    """
    RunPod handler. Forwards the job input directly to vLLM's
    OpenAI-compatible chat completions endpoint.

    Expected input format (same as OpenAI chat completions):
    {
        "model": "zai-org/GLM-OCR",
        "messages": [...],
        "max_tokens": 2048,
        "temperature": 0.0
    }
    """
    job_id = job.get("id", "unknown")
    job_input = job.get("input")
    if not isinstance(job_input, dict):
        return {
            "error": (
                "Invalid request format. Expected {'input': {...}} as the "
                "job payload."
            )
        }

    job_input = dict(job_input)
    log.info("Job %s: received request", job_id)

    # Set model default if not provided.
    if "model" not in job_input:
        job_input["model"] = MODEL_NAME

    # vLLM chat completions requires messages.
    if "messages" not in job_input:
        log.error("Job %s: missing required 'messages' field", job_id)
        return {
            "error": "Input must contain a 'messages' field for Chat Completions API."
        }

    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=job_input,
            timeout=600,
        )

        if response.status_code != 200:
            log.error(
                "Job %s: vLLM returned %s with body: %s",
                job_id,
                response.status_code,
                response.text,
            )

        response.raise_for_status()
        result = response.json()
        log.info("Job %s: completed", job_id)
        return result
    except requests.exceptions.RequestException as exc:
        detail = ""
        if getattr(exc, "response", None) is not None:
            detail = f" | response_body={exc.response.text}"
        log.error("Job %s: failed - %s%s", job_id, exc, detail)
        raise


if __name__ == "__main__":
    vllm_process = start_vllm()
    wait_for_vllm()
    runpod.serverless.start({"handler": handler})
