#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}
norms=(L2 Linf GDPR_CFE SCFE)
seeds=(10)
seed=${seeds[0]}

for norm in "${norms[@]}"; do
        python3 natural_celebA.py $norm $seed $devices hyperparams/hyperparams_celeba.json >> logs/${now}.out 2>&1
done
