#!/bin/bash

set -e

for idx in 2471 1586 1275 2646 2712 280 664 1777 580 503; do
    tmp=$(mktemp /tmp/shapley_XXXX.yaml)
    sed "s/^TARGET_INDEX: .*/TARGET_INDEX: ${idx}/" configs/shapley.yaml > "$tmp"
    sed -i "s/^MLFLOW_RUN_NAME: .*/MLFLOW_RUN_NAME: \"target_${idx}_male_N1_tau_0.5_nfe_20\"/" "$tmp"
    python3 -m scripts.run_shapley --config "$tmp"
    rm "$tmp"
done

