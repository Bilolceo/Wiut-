# Fallback runtime for the two official commands on the eval machine (T4, CUDA).
# libgl1/libglib2.0-0: ultralytics pulls in opencv-python, whose cv2 build links
# against them; without these `import cv2` fails on minimal images.
FROM python:3.11-slim
# torch==2.14.0 from PyPI ships its own CUDA runtime libraries (T4 driver on the host).

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN test -f weights/yolo11n.pt || ./weights/download.sh
ENV YOLO_OFFLINE=1

CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "predictions.json"]
