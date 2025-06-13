#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}

datasets=(CIFAR10)
modes=(natural_rand natural_det)
norms=(SCFE)

for dataset in "${datasets[@]}"; do
for mode in "${modes[@]}"; do
for norm in "${norms[@]}"; do
  python3 create.py $dataset $mode $norm $devices >> logs/${now}.out 2>&1
done
done
done

: '
for norm in "${norms[@]}"; do
  python3 create.py CIFAR10 uniform $norm $devices --large_epsilon >> logs/${now}.out 2>&1
  python3 create.py CIFAR10 uniform_sub $norm $devices >> logs/${now}.out 2>&1
done
'
