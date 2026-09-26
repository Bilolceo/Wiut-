# Stub: fallback runtime for the two official commands.
# TODO: switch to a CUDA base image (T4, fp16) once torch is added to requirements.txt.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "predictions.json"]
