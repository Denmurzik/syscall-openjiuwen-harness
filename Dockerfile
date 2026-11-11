FROM python:3.11.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the harness package and its entry point go into the image (see .dockerignore).
COPY harness/ harness/
COPY run_agent.py .

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# The jury refuses a container that reaches the internet. The harness itself only talks to LLM_BASE_URL; these keep the
# libraries pulled in by openjiuwen (huggingface_hub, transformers, opentelemetry) from ever trying a download or an export.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    OTEL_SDK_DISABLED=1 OTEL_TRACES_EXPORTER=none OTEL_METRICS_EXPORTER=none OTEL_LOGS_EXPORTER=none DO_NOT_TRACK=1

# No USER on purpose: the owner of the mounted /data/tasks is not known in advance, so a non-root user
# could not write answers there. The run has no capabilities (jury flags), so root has no special rights.
# Запуск задаётся командой жюри (--entrypoint python); CMD только для ручного запуска.
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
