# Docker image for running StegaGen on Hugging Face Spaces (Docker SDK).
FROM python:3.12-slim

# System libraries needed by OpenCV (used inside ultralytics).
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face runs the container as user id 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config/Ultralytics
WORKDIR $HOME/app

# CPU-only PyTorch first (much smaller than the default GPU build),
# then the rest of the requirements.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Download the YOLO detector once, at build time, so the app starts fast
# and the 19 MB model file does not have to be uploaded to the Space.
RUN python -c "from ultralytics import YOLO; YOLO('yolo11s.pt')"

COPY --chown=user app.py stego_core.py ./

# Self-test during the build: the build fails if the algorithm is broken.
RUN python stego_core.py

EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
