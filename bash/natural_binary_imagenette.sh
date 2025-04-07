#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}
norms=(GDPR_CFE SCFE L2 Linf)
mode=(det)
seeds=(10)
seed=${seeds[0]}

for mode in "${mode[@]}"; do
    for norm in "${norms[@]}"; do
        python3 natural_binary_imagenette.py $norm $mode $seed $devices >> logs/${now}.out 2>&1
    done
done
