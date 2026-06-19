#!/bin/bash

set -e

for top_k in 1 2 3; do
    for seed in 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 42 43 44 45 46 47 48 49 59 60 61 62 63 64 65 66 67 68 69 70; do
        tmp=$(mktemp /tmp/retrain_XXXX.yaml)
        sed "s/^SEED: .*/SEED: ${seed}/" configs/retrain.yaml > "$tmp"
        sed -i "s/^TOP_K: .*/TOP_K: ${top_k}/" "$tmp"
        sed -i "s/^TRAINING_RUN_NAME: .*/TRAINING_RUN_NAME: \"i2sb_tau_0.5_topk_${top_k}\"/" "$tmp"
        python3 -m src.retraining.retrain --config "$tmp"
        rm "$tmp"
    done
done

