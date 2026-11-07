FROM python:3.11.11-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY harness/ harness/
COPY run_agent.py .

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Запуск задаётся командой жюри (--entrypoint python); CMD только для ручного запуска.
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
