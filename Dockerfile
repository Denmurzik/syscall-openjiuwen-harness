FROM python:3.11.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the harness package and its entry point go into the image (see .dockerignore).
COPY harness/ harness/
COPY run_agent.py .

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# No USER on purpose: the owner of the mounted /data/tasks is not known in advance, so a non-root user
# could not write answers there. The run has no capabilities (jury flags), so root has no special rights.
# Запуск задаётся командой жюри (--entrypoint python); CMD только для ручного запуска.
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
