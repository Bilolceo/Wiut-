# Live demo (API + website) for a CPU host such as a Hugging Face Space (Docker SDK, port 7860).
#   docker build -f demo/Dockerfile -t wiut-demo . && docker run -p 7860:7860 wiut-demo
# The Qwen3-VL verifier is optional here: set WITH_VLM=1 at build time to bake in the 4.3 GB weights
# (otherwise "full analysis" simply skips the VLM classes).
FROM python:3.11-slim
ARG WITH_VLM=0
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
COPY demo/requirements.txt demo/
RUN pip install --no-cache-dir -r demo/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
COPY . .
RUN if [ "$WITH_VLM" = "1" ]; then bash weights/download.sh; fi
ENV YOLO_OFFLINE=1 WIUT_DEVICE=cpu DEMO_WORK_DIR=/tmp/wiut-demo
EXPOSE 7860
CMD ["uvicorn", "demo.server:app", "--host", "0.0.0.0", "--port", "7860"]
