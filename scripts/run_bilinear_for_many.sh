#!/bin/bash

set -e

for idx in 2471 1586 1275 2646 2712 1050 933 1242 497 2606 1855 429 942 2813 1865 1745 173 1552 2356 2683 2692 622 2217 1258 2189 137 988 1622 2781 447 1909 575 1982 792 2451 2155 1185 386 804 2696; do
    tmp=$(mktemp /tmp/bilinear_XXXX.yaml)
    sed "s/^TARGET_INDEX: .*/TARGET_INDEX: ${idx}/" configs/bilinear.yaml > "$tmp"
    sed -i "s/^MLFLOW_RUN_NAME: .*/MLFLOW_RUN_NAME: \"final_${idx}_blackfill\"/" "$tmp"
    python3 -m scripts.run_bilinear --config "$tmp"
    rm "$tmp"
done