#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}
norms=(SCFE GDPR_CFE L2 Linf)
mode=(det)
seeds=(10)
seed=${seeds[0]}

for mode in "${mode[@]}"; do
    for norm in "${norms[@]}"; do
        python3 natural_waterbirds.py $norm $mode $seed $devices >> logs/${now}.out 2>&1
    done
done
