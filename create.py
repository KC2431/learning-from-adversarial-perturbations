import argparse
import os
from collections import OrderedDict
from typing import Any, Dict, Literal

import torch
from pytorch_lightning.lite import LightningLite
from torchvision.transforms import ToTensor

from utils.classifiers import ConvNet, WideResNet
from utils.create import Create
from utils.datasets import CIFAR10, FMNIST, MNIST
from utils.utils import ModelWithNormalization, dataloader, set_seed


class Main(LightningLite):
    def run(
        self,
        dataset_name: Literal['MNIST', 'FMNIST', 'CIFAR10'],
        mode: Literal['natural_rand', 'natural_det', 'uniform'], #, 'uniform_sub'],
        norm: Literal['L0', 'L2', 'Linf'],
        #large_epsilon: bool = False,
    ) -> None:
        
        root = os.path.join('datasets', f'{dataset_name}_{mode}_{norm}')
        #root = root + '_large' if large_epsilon else root

        if os.path.exists(root):
            print(f'already exist: {root}')
            return
        else:
            #self.barrier()
            #if self.is_global_zero:
            os.makedirs(root)

        dataset_root = os.path.join(os.path.sep, 'root', 'datasets')

        ckpt_dir_path = os.path.join('models', dataset_name, 'version_0', 'checkpoints')
        ckpt_name = [fname for fname in os.listdir(ckpt_dir_path) if '.ckpt' in fname][0]
        ckpt_path = os.path.join(ckpt_dir_path, ckpt_name)

        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        state_dict = OrderedDict((k.replace('classifier.', ''), v) for k, v in state_dict.items())

        if dataset_name in ('MNIST', 'FMNIST'):
            dataset_cls = MNIST if dataset_name == 'MNIST' else FMNIST
            batch_size = 100000
            total_samples = 1200000

        elif dataset_name == 'CIFAR10':
            dataset_cls = CIFAR10
            batch_size = 2500
            total_samples = 50000
            #target_classes = [3, 9] if norm == 'L2' else [0, 3]

        else:
            raise ValueError(dataset_name)
        
        n_class = dataset_cls.n_class
        size = dataset_cls.size

        atk_kwargs: Dict[str, Any] = {'norm': norm}
        
        if mode in ('natural_rand', 'natural_det'):
            dataset = dataset_cls(dataset_root, True, ToTensor())
            loader = dataloader(dataset, batch_size, False)
            loader = self.setup_dataloaders(loader)

        if dataset_name == 'MNIST':
            #assert not large_epsilon
            classifier = ConvNet(n_class)
            atk_kwargs['steps'] = 100 if norm in ('L2', 'Linf') else 10
            atk_kwargs['eps'] = 2 if norm == 'L2' else 0.3 if norm == 'Linf' else None

        elif dataset_name == 'FMNIST':
            #assert not large_epsilon
            classifier = ConvNet(n_class)
            atk_kwargs['steps'] = 100 if norm in ('L2', 'Linf') else 35
            atk_kwargs['eps'] = 2 if norm == 'L2' else 0.3 if norm == 'Linf' else None

        elif dataset_name == 'CIFAR10':
            classifier = WideResNet(28, 10, 0.3, n_class)
            #if large_epsilon:
            #    atk_kwargs['steps'] = 100 if norm == 'L2' else 460
            #    atk_kwargs['eps'] = 1.06 if norm == 'L2' else None
            #else:
            atk_kwargs['steps'] = 100 if norm in ('L2', 'Linf') else 150
            atk_kwargs['eps'] = 0.5 if norm == 'L2' else 0.1 if norm == 'Linf' else None

        else:
            raise ValueError(dataset_name)

        classifier = ModelWithNormalization(classifier, dataset_cls.mean, dataset_cls.std)
        classifier.load_state_dict(state_dict)
        classifier = self.setup(classifier)
        
        #set_seed(self.global_rank)
        set_seed()

        create = Create(classifier, atk_kwargs, root) #, self.global_rank)

        if mode in ['natural_rand', 'natural_det']:
            create.natural(loader, n_class, mode.replace('natural_', '')) # type: ignore

        elif mode == 'uniform':
            create.uniform(batch_size, total_samples, n_class, size) # type: ignore

        #elif mode == 'uniform_sub':
        #    create.uniform_sub(batch_size, total_samples, target_classes, size) # type: ignore

        else:
            raise ValueError(mode)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name', choices=('MNIST', 'FMNIST', 'CIFAR10'))
    parser.add_argument('mode', choices=(
        'natural_rand', 
        'natural_det', 
        'uniform', 
        #'uniform_sub',
    ))
    parser.add_argument('norm', choices=('L0', 'L2', 'Linf'))
    parser.add_argument('devices', nargs='+', type=int)
    #parser.add_argument('--large_epsilon', action='store_true')
    args = parser.parse_args()

    lite_kwargs = {
        'accelerator': 'gpu',
        'strategy': 'ddp_find_unused_parameters_false',
        'devices': args.devices,
        'precision': 16,
    }
    
    #Main(**lite_kwargs).run(args.dataset_name, args.mode, args.norm, args.large_epsilon)
    Main(**lite_kwargs).run(args.dataset_name, args.mode, args.norm)