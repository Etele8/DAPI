#!/usr/bin/env bash
# Default Cellpose-SAM fine-tuning launcher for DAPI bacterial-cell crops.
#
# Usage:
#   bash /opt/runpod_train.sh                          # uses /workspace/cellpose_data
#   bash /opt/runpod_train.sh /path/to/data            # explicit data dir
#   bash /opt/runpod_train.sh /path/to/data --n_epochs 50 --learning_rate 5e-6
#
# Anything after the data directory is forwarded verbatim to `python -m cellpose`,
# so you can override any flag without editing this script.

set -euo pipefail

DATA_DIR="${1:-/workspace/cellpose_data}"
shift || true

if [ ! -d "$DATA_DIR/train" ] || [ ! -d "$DATA_DIR/val" ]; then
    echo "Expected $DATA_DIR/train and $DATA_DIR/val to exist." >&2
    echo "Got contents:" >&2
    ls -la "$DATA_DIR" 2>&1 >&2 || true
    exit 1
fi

echo "Training Cellpose-SAM on $DATA_DIR"
echo "  train: $(ls "$DATA_DIR/train" | grep -v '_masks' | wc -l) images"
echo "  val  : $(ls "$DATA_DIR/val" | grep -v '_masks' | wc -l) images"

exec python -m cellpose --train \
    --dir "$DATA_DIR/train" \
    --test_dir "$DATA_DIR/val" \
    --pretrained_model cyto3 \
    --mask_filter _masks \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --n_epochs 40 \
    --save_every 5 \
    --batch_size 8 \
    --min_train_masks 1 \
    --use_gpu \
    --verbose \
    "$@"
