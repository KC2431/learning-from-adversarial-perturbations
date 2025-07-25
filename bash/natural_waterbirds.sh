#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}
norms=(L2 Linf GDPR_CFE SCFE)
comb_nat_perts=(yes no)
seeds=(10 20 40)
percentages=(20 50)

for comb_nat_pert in "${comb_nat_perts[@]}"; do
        for seed in "${seeds[@]}"; do
                for percentage in "${percentages[@]}"; do
                        for norm in "${norms[@]}"; do
                                python3 natural_waterbirds.py $norm $comb_nat_pert $percentage $seed $devices >> logs/${now}.out 2>&1
                        done
                done
        done
done
