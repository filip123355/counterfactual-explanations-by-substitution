#!/bin/bash

set -e

for top_k in 1 2 3; do
    for seed in 42 43 44 45 46; do
        tmp=$(mktemp /tmp/retrain_XXXX.yaml)
        sed "s/^SEED: .*/SEED: ${seed}/" configs/retrain.yaml > "$tmp"
        sed -i "s/^TOP_K: .*/TOP_K: ${top_k}/" "$tmp"
        python3 -m src.retraining.retrain --config "$tmp"
        rm "$tmp"
    done
done

