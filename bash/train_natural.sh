#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}

datasets=(MNIST FMNIST CIFAR10)
for dataset in "${datasets[@]}"; do
  python3 train.py $dataset $devices >> logs/${now}.out 2>&1
done