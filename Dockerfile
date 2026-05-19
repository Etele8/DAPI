# Cellpose-SAM training image for DAPI bacterial-cell crops, sized for RunPod.
#
# Bakes:
#   - PyTorch 2.4 / CUDA 12.4 (from runpod/pytorch base)
#   - Cellpose 4.x
#   - The cpsam transformer weights (~4 GB) pre-downloaded into /root/.cellpose
#   - Small dev quality-of-life utilities (tmux, htop, unzip, vim-tiny)
#   - A one-line training launcher at /opt/runpod_train.sh
#
# Build locally (from project root):
#   docker build -t <dockerhub-user>/dapi-cellpose:latest .
#
# Push to Docker Hub:
#   docker login
#   docker push <dockerhub-user>/dapi-cellpose:latest
#
# Then on RunPod -> Templates -> New Template:
#   Container Image: <dockerhub-user>/dapi-cellpose:latest
#   Container Disk : 30 GB
#   Volume Disk    : 20 GB (optional, persists training data across pods)
#   Volume Path    : /workspace
#   Expose HTTP    : (none needed)
#
# Spin up a pod from the template, scp the cellpose_data.zip into /workspace,
# then on the pod:
#   unzip cellpose_data.zip -d cellpose_data
#   bash /opt/runpod_train.sh cellpose_data
#
# The trained model lands in cellpose_data/train/models/. Download it with scp.

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        tmux \
        htop \
        unzip \
        less \
        vim-tiny \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir cellpose

RUN mkdir -p /root/.cellpose/models \
    && curl -L --fail \
        -o /root/.cellpose/models/cpsam \
        https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam

COPY docker/runpod_train.sh /opt/runpod_train.sh
RUN chmod +x /opt/runpod_train.sh

WORKDIR /workspace

CMD ["/bin/bash"]
