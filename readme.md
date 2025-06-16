This is a forked and adapted version of the official code for "Theoretical Understanding of Learning from Adversarial Perturbations" [S. Kumano et al., ICLR24].

All generated data, including artificial datasets, model parameters, and adversarial examples, can be downloaded from [here](https://drive.google.com/file/d/1gcS_sBp65zwl5yS5gCg5S884tvH1KwKi/view) or [here](https://filedn.com/lAlreeY65CBjFVbAkaD5F7k/Research/%5BICLR24%5D%20Theoretical%20Understanding%20of%20Learning%20from%20Adversarial%20Perturbations/data.zip).

# Run
```console
bash/artificial.sh <gpu_id: int>
bash/train_natural.sh <gpu_id: int>
bash/create.sh <gpu_id: int>
bash/train_perturbation.sh <gpu_id: int>
```

# Run
```console
bash/natural_waterbirds.sh <gpu_id: int>
bash/natural_spuco_dogs.sh <gpu_id: int>
bash/natural_chexpert.sh <gpu_id: int>
bash/natural_celeba.sh <gpu_id: int>
```
