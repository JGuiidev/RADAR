#!/usr/bin/env bash
set -euo pipefail
python train_model.py --dataset Kvasir "$@"
