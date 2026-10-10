#!/usr/bin/env bash
set -euo pipefail
python eval_model.py --dataset Kvasir "$@"
