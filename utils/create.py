import os
import random
from typing import Any, Dict, Literal, Tuple
import sys

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from .attacks import PGDL2, PGDLinf
from .utils import ModelWithNormalization, get_model_device
from .scfe import APG0_CFE
from .gdpr_cfe import GDPR_CFE
import sys

class Create:
    def __init__(
        self, 
        classifier: ModelWithNormalization, 
        atk_kwargs: Dict[str, Any], 
        root: str, 
        #rank: int,
    ) -> None:
        self.classifier = classifier
        self.atk_kwargs = atk_kwargs

        if atk_kwargs['norm'] == 'GDPR_CFE':
            self.atk= GDPR_CFE(
                model=classifier,
                max_image_range = 1.0,
                min_image_range = 0.0, 
                optimizer = torch.optim.Adam, 
                iters=atk_kwargs['steps'], 
                lamb=atk_kwargs['lamb_gdpr'],
                lamb_cf=atk_kwargs['lamb_cf_gdpr'],
                device= 'cuda:0',
                mode='natural'
            )
        elif atk_kwargs['norm'] == 'SCFE':
            self.atk = APG0_CFE
        elif atk_kwargs['norm'] == 'L2':
            self.atk = PGDL2(classifier, atk_kwargs['steps'], atk_kwargs['eps'])
        elif atk_kwargs['norm'] == 'Linf':
            self.atk = PGDLinf(classifier, atk_kwargs['steps'], atk_kwargs['eps'])
        else:
            raise ValueError(atk_kwargs['norm'])
        
        self.root = root
        #self.rank = rank

        self.device = get_model_device(classifier)
        self.dataset = {'imgs': [], 'labels': []}
    

    def _generate_and_save_advs(self, imgs: Tensor, target_labels: Tensor) -> None:
        if self.atk_kwargs['norm'] == 'SCFE':
            imgs_flattened = imgs.view(imgs.size(0), -1)

            maxs = imgs_flattened.max().repeat(imgs_flattened.size(-1))
            mins = imgs_flattened.min().repeat(imgs_flattened.size(-1))
            
            del imgs_flattened

            attack = self.atk(model=self.classifier, 
                              mins=mins, 
                              maxs=maxs,
                              range_max=None,
                              range_min=None,
                              lam_steps=self.atk_kwargs['lamb_steps_scfe'],
                              scale_model=False,
                              iters=100,
                              numclasses=10,
                              L0=self.atk_kwargs['L0_scfe'],
                    )

            advs = attack.get_CFs(imgs, target_labels.unsqueeze(1))
        elif self.atk_kwargs['norm'] == 'GDPR_CFE':
            self.atk.min_image_range = imgs.min()
            self.atk.max_image_range = imgs.max()
            advs = self.atk.get_perturbations(imgs, target_labels.unsqueeze(1))
        else:
            advs = self.atk(imgs, target_labels)
        advs = advs.cpu()
        target_labels = target_labels.cpu()

        self.dataset['imgs'].append(advs)
        self.dataset['labels'].append(target_labels)


    def _save_dataset(self) -> None:
        dataset = {}
        dataset['imgs'] = torch.cat(self.dataset['imgs']) # type: ignore
        dataset['labels'] = torch.cat(self.dataset['labels']) # type: ignore

        fname = 'dataset' #if self.rank == 0 else f'dataset_{self.rank}'
        path = os.path.join(self.root, fname)
        torch.save(dataset, path)


    @staticmethod
    def _select_other_labels(labels: Tensor, n_class: int) -> Tensor:
        torch_range = torch.arange(start=0, end=n_class)
        choices = [torch_range[torch_range!=i] for i in range(n_class)]
        others = map(lambda label: choices[label][random.randint(0, n_class-2)], labels)
        return torch.tensor(list(others), device=labels.device)


    def natural(self, dataloader: DataLoader, n_class: int, mode: Literal['rand', 'det']) -> None:

        orig_imgs = []

        for imgs, labels in tqdm(dataloader):
            if mode == 'rand':
                assert n_class > 2
                target_labels = self._select_other_labels(labels, n_class)
            elif mode == 'det':
                target_labels = (labels + 1) % n_class
            else:
                raise ValueError(mode)
            orig_imgs.append(imgs)
            self._generate_and_save_advs(imgs, target_labels)
        
        
        
        self._save_dataset()
        
        orig_imgs = torch.cat(orig_imgs)
        adv_data = torch.load(f'/home/htc/kchitranshi/SCRATCH/CFE_datasets/CIFAR10_natural_{mode}_{self.atk_kwargs["norm"]}/dataset')['imgs']
        print(f'Average L2 norm across samples {(adv_data.cpu() - orig_imgs.cpu()).norm(p=2,dim=(1,2,3)).mean()}')
        sys.exit()
        

    def uniform(
        self, 
        batch_size: int, 
        total_samples: int, 
        n_class: int, 
        img_size: Tuple[int, ...],
    ) -> None:
        assert total_samples % batch_size == 0
        #orig_imgs = []
        for _ in tqdm(range(total_samples // batch_size)):
            
            imgs = torch.rand(batch_size, *img_size, device=self.device)
            target_labels = torch.randint(0, n_class, (batch_size,), device=self.device)
            #orig_imgs.append(imgs)
            self._generate_and_save_advs(imgs, target_labels)
        """
        orig_imgs = torch.cat(orig_imgs)
        adv_data = torch.load(f'/home/htc/kchitranshi/SCRATCH/temp_CFE_datasets/MNIST_uniform_{self.atk_kwargs["norm"]}/dataset')['imgs']
        print(f'Average L2 norm across samples {(adv_data.cpu() - orig_imgs.cpu()).norm(p=2,dim=(1,2,3)).mean()}')
        sys.exit()
        """
        self._save_dataset()


    '''
    from typing import List
    
    def uniform_sub(
        self, 
        batch_size: int, 
        total_samples: int, 
        target_classes: List[int], 
        img_size: Tuple[int, ...],
    ) -> None:
        assert total_samples % batch_size == 0

        target_classes_tensor = torch.tensor(target_classes, device=self.device)

        for _ in tqdm(range(total_samples // batch_size)):
            
            imgs = torch.rand(batch_size, *img_size, device=self.device)
            indices = torch.randint(0, len(target_classes), (batch_size,))
            target_labels = target_classes_tensor[indices]

            self._generate_and_save_advs(imgs, target_labels)

        self._save_dataset()
    '''
