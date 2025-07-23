This is a forked and adapted version of the official code for "Theoretical Understanding of Learning from Adversarial Perturbations" [S. Kumano et al., ICLR24].

To create adversarial attacks/Counterfactuals on the WaterBirds dataset [Sagawa et al., 2020a].

# Run 
```console
bash/natural_waterbirds.sh <gpu_id: int>
```
---
To create adversarial attacks/Counterfactuals on the SpuCo dogs dataset [Joshi et al., 2023].

# Run
```console
bash/natural_spuco_dogs.sh <gpu_id: int>
```
---
To create adversarial attacks/Counterfactuals on the CelebA dataset [Liang & Zou, 2022]. 

# Run
```console
bash/natural_celebA.sh <gpu_id: int>
```
---
To create adversarial attacks/Counterfactuals on the ChestXpert dataset [Pavasovic et al., 2025].

# Run 
```console
bash/natural_chexpert.sh <gpu_id: int>
```
---
To get the worst group accuracy and standard accuracy on any of the aforementioned datasets.

# Run
```console
python worst_group_accuracy.py \
       --dataset SpuCO_dogs \ 
       --split test \
       --show_normal_acc yes \
       --comb_nat_pert no \
       --percentage 20
```

Here, the available datasets are `SpuCo_dogs, WaterBirds, cheXpert, celebA`. The `--split` argument allows to choose between `train, val` and `test` splits of any dataset. The `--show_normal_acc` argument is for displaying standard accuracy on a split. 

# Run
```console
bash/artificial.sh <gpu_id: int>
bash/train_natural.sh <gpu_id: int>
bash/create.sh <gpu_id: int>
bash/train_perturbation.sh <gpu_id: int>
```


