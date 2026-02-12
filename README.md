# GLM-OCR Docker Image for RunPod Serverless

Docker image for running [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (0.9B parameter OCR model) on [RunPod Serverless](https://www.runpod.io/serverless-gpu) using vLLM.

Model weights are baked into the image at build time for fast cold starts.

## What's included

- **Base image:** `vllm/vllm-openai:nightly`
- **Model:** [zai-org/GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) (MIT License)
- **Transformers:** v5+ dev branch (required by GLM-OCR)
- **Serving:** vLLM on port 8080

## Deploy on RunPod Serverless

1. Create a new Serverless endpoint on RunPod.
2. Select **Build from GitHub repo** and point it to this repository.
3. No container start command is needed — the `CMD` in the Dockerfile handles it.
4. (Optional) Set `HF_TOKEN` as an environment variable in RunPod's UI for faster model downloads during builds.

## Usage

GLM-OCR supports two prompt types:

### Document parsing

Extract raw content from documents using these prompts:

| Task    | Prompt                 |
|---------|------------------------|
| Text    | `Text Recognition:`    |
| Formula | `Formula Recognition:` |
| Table   | `Table Recognition:`   |

### Information extraction

Extract structured data by providing a JSON schema as the prompt. Example:

```
Please output the information in the image in the following JSON format:
{
    "name": "",
    "date": "",
    "total": ""
}
```

### API example

Once deployed, the endpoint serves an OpenAI-compatible API:

```bash
curl http://<your-endpoint>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "zai-org/GLM-OCR",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/document.png"}},
          {"type": "text", "text": "Text Recognition:"}
        ]
      }
    ]
  }'
```

## Build locally

```bash
docker build -t glm-ocr .
docker run --gpus all -p 8080:8080 glm-ocr
```

## License

This Dockerfile is provided as-is. GLM-OCR is released under the [MIT License](https://huggingface.co/zai-org/GLM-OCR). The vLLM base image has its own license terms.
