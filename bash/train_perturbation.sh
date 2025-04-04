#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}

datasets=(FMNIST MNIST CIFAR10)
modes=(natural_rand natural_det)
norms=(L2 Linf)
for dataset in "${datasets[@]}"; do
for mode in "${modes[@]}"; do
for norm in "${norms[@]}"; do
  python3 train.py ${dataset}_${mode}_${norm} $devices >> logs/${now}.out 2>&1
done
done
done

datasets=(MNIST FMNIST)
ratios=(0.05 0.2 0.4 0.6 0.8 1.0)
for dataset in "${datasets[@]}"; do
for ratio in "${ratios[@]}"; do
for norm in "${norms[@]}"; do
  python3 train.py ${dataset}_uniform_${norm}_${ratio} $devices >> logs/${now}.out 2>&1
done
done
done

for norm in "${norms[@]}"; do
  python3 train.py CIFAR10_uniform_${norm} $devices >> logs/${now}.out 2>&1
  # python3 train.py CIFAR10_uniform_${norm}_large $devices >> logs/${now}.out 2>&1
  # python3 train.py CIFAR10_uniform_sub_${norm} $devices >> logs/${now}.out 2>&1
done
