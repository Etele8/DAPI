# Cellpose-SAM training image for DAPI bacterial-cell crops, sized for RunPod.
#
# Bakes:
#   - PyTorch 2.4 / CUDA 12.4 (from runpod/pytorch base)
#   - Cellpose 4.x + cpsam transformer weights (~4 GB) at /root/.cellpose
#   - SSH daemon configured for RunPod's PUBLIC_KEY env var injection
#   - tmux, htop, unzip, rsync, vim-tiny for dev quality-of-life
#   - Training launcher at /opt/runpod_train.sh
#
# Build (from project root):
#   docker build -t <dockerhub-user>/dapi-cellpose:latest .
#
# Push:
#   docker login
#   docker push <dockerhub-user>/dapi-cellpose:latest
#
# RunPod -> Templates -> New Template:
#   Container Image     : <dockerhub-user>/dapi-cellpose:latest
#   Container Disk      : 30 GB
#   Volume Disk         : 20 GB (optional; persists data across pod restarts)
#   Volume Mount Path   : /workspace
#   Expose TCP Ports    : 22         <-- required for SSH/scp/rsync
#   Container Start Cmd : (leave empty; image's CMD handles startup)
#
# In RunPod account settings, paste your SSH public key once. RunPod injects it
# into each pod via the PUBLIC_KEY env var, and the image's start.sh writes it
# to /root/.ssh/authorized_keys before launching sshd.
#
# Per-run workflow:
#   1. Spin up pod from template
#   2. From RunPod's pod page, copy the SSH command (ssh root@<host> -p <port>)
#   3. Local: `scp cellpose_data.zip root@<host>:/workspace/ -P <port>`
#      (or `rsync -avh ...` for resumable / incremental transfers)
#   4. SSH in: `unzip cellpose_data.zip -d cellpose_data && bash /opt/runpod_train.sh cellpose_data`
#   5. Trained model lands in cellpose_data/train/models/; pull it back with scp.

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
        openssh-server \
        rsync \
    && rm -rf /var/lib/apt/lists/*

# SSH config: permit key-based root login (the RunPod convention).
RUN mkdir -p /var/run/sshd \
    && sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#*PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config

RUN pip install --no-cache-dir cellpose

RUN mkdir -p /root/.cellpose/models \
    && curl -L --fail \
        -o /root/.cellpose/models/cpsam \
        https://huggingface.co/mouseland/cellpose-sam/resolve/main/cpsam

COPY docker/runpod_train.sh /opt/runpod_train.sh
COPY docker/start.sh /opt/start.sh
RUN chmod +x /opt/runpod_train.sh /opt/start.sh

EXPOSE 22

WORKDIR /workspace

CMD ["/opt/start.sh"]
