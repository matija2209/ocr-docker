"""
RunPod Serverless handler for GLM-OCR via vLLM.

Starts vLLM HTTP server in the background, waits for it to be ready,
then forwards incoming RunPod jobs to the local OpenAI-compatible API.
"""

import os
import time
import subprocess
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


def start_vllm():
    """Start vLLM as a background process."""
    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--allowed-local-media-path", "/",
        "--port", str(VLLM_PORT),
        "--max-model-len", MAX_MODEL_LEN,
        "--speculative-config.method", "mtp",
        "--speculative-config.num_speculative_tokens", "1",
    ]
    if ENFORCE_EAGER:
        cmd.append("--enforce-eager")
    log.info(f"Starting vLLM: {' '.join(cmd)}")
    process = subprocess.Popen(cmd)
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
    log.info(f"Job {job_id}: received request")

    # Set model default if not provided
    if "model" not in job_input:
        job_input["model"] = MODEL_NAME

    try:
        response = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json=job_input,
            timeout=600,
        )
        response.raise_for_status()
        result = response.json()
        log.info(f"Job {job_id}: completed")
        return result
    except requests.exceptions.RequestException as e:
        log.error(f"Job {job_id}: failed - {e}")
        raise


if __name__ == "__main__":
    vllm_process = start_vllm()
    wait_for_vllm()
    runpod.serverless.start({"handler": handler})
