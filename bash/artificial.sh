#!/bin/bash

set -eux

mkdir -p logs

now=$(TZ=UTC-9 date '+%Y-%m-%d-%H-%M-%S')

devices=${@}

hidden_dim=1000
n_sample=1000
n_noise_sample=10000

# Best seed for visualizations of decision maps.
# This does not change any tendency of artificial experiments.
# We select seeds from 0--6
mode_and_seed_list=("uniform 5" "gauss 2")
for mode_and_seed in "${mode_and_seed_list[@]}"; do
  s=($mode_and_seed)
  mode=${s[0]}
  seed=${s[1]}

  ##################################################################
  norm=L0
  perturbation_constraint=0.05

  in_dims=(100 500 1000 2500 5000 7500 10000)
  for in_dim in "${in_dims[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  in_dim=10000

  n_samples=(2000 3000 5000 7000 10000)
  for n_sample in "${n_samples[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  n_sample=1000

  n_noise_samples=(1 10 100 500 1000 1500 2500 5000 7500)
  for n_noise_sample in "${n_noise_samples[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  n_noise_sample=10000

  perturbation_constraints=(0.0001 0.0004 0.0008 0.001 0.002 0.004 0.006 0.008 0.01 0.02 0.03 0.04 0.05)
  for perturbation_constraint in "${perturbation_constraints[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  perturbation_constraint=0.05

  ##################################################################
  norm=L2
  perturbation_constraint=0.78

  in_dim_and_perturbation_constraint_list=("100 0.078" "500 0.17" "1000 0.24" "2500 0.39" "5000 0.55" "7500 0.67" "10000 0.78")
  for in_dim_and_perturbation_constraint in "${in_dim_and_perturbation_constraint_list[@]}"; do
    s=($in_dim_and_perturbation_constraint)
    in_dim=${s[0]}
    perturbation_constraint=${s[1]}
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  in_dim=10000
  perturbation_constraint=0.78

  n_samples=(2000 3000 5000 7000 10000)
  for n_sample in "${n_samples[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  n_sample=1000

  n_noise_samples=(1 10 100 500 1000 1500 2500 5000 7500)
  for n_noise_sample in "${n_noise_samples[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  n_noise_sample=10000

  perturbation_constraints=(0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6)
  for perturbation_constraint in "${perturbation_constraints[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  perturbation_constraint=0.78

  ##################################################################
  norm=Linf
  perturbation_constraint=0.03

  in_dims=(100 500 1000 2500 5000 7500 10000)
  for in_dim in "${in_dims[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  in_dim=10000

  n_samples=(2000 3000 5000 7000 10000)
  for n_sample in "${n_samples[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  n_sample=1000

  n_noise_samples=(1 10 100 500 1000 1500 2500 5000 7500)
  for n_noise_sample in "${n_noise_samples[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  n_noise_sample=10000

  perturbation_constraints=(0.001 0.005 0.01 0.015 0.02)
  for perturbation_constraint in "${perturbation_constraints[@]}"; do
    python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
  done
  perturbation_constraint=0.03

done

n_noise_sample=20
norm=L2
mode=uniform
perturbation_constraint=0.78
seed=5
python3 artificial.py $in_dim $hidden_dim $n_sample $n_noise_sample $norm False $mode $perturbation_constraint $seed $devices >> logs/${now}.out 2>&1
