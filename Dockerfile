FROM vllm/vllm-openai:nightly

# Install newer Transformers so GLM-OCR is recognized
RUN pip uninstall -y transformers || true \
 && pip install -U git+https://github.com/huggingface/transformers.git

ENV HF_HOME=/root/.cache/huggingface
