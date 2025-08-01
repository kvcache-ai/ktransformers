#!/bin/bash
set -e

MODE=$1

if [[ "$MODE" != "fast" && "$MODE" != "full" ]]; then
  echo "Usage: $0 [fast|full]"
  exit 1
fi

source ./set_model_paths.sh

echo "Running in $MODE mode..."

# Figure 11 - prefill - fast ~40min | full ~5h
echo "===> Running Figure11-prefill ($MODE)..."
cd Figure11-prefill/${MODE}_run
bash run_ktransformers.sh
bash run_fiddler.sh
bash run_llama.cpp.sh
python ../plot.py
cd ../../

# Figure 12 - decode - fast ~50min | full ~4h
echo "===> Running Figure12-decode ($MODE)..."
cd Figure12-decode/${MODE}_run
bash run_ktransformers.sh
bash run_fiddler.sh
bash run_llama.cpp.sh
python ../plot.py
cd ../../

# Figure 13 - breakdown - fast ~40min | full ~6h
echo "===> Running Figure13-breakdown ($MODE)..."
cd Figure13-breakdown/${MODE}_run
bash run_prefill.sh
bash run_decode.sh
python ../plot.py
cd ../../

# Table 2 - accuracy - fast ~50min | full ~45h
echo "===> Running Table2-accuracy ($MODE)..."
cd Table2-accuracy/${MODE}_run
bash run_eval.sh
bash run_scoring.sh
python ../plot.py
cd ../../

echo "✅ All $MODE runs completed successfully."