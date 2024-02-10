import argparse
import os
from typing import List

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.csv_logs import CSVLogger
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms as T

from utils.callbacks import EpochProgressBar
from utils.classifiers import ConvNet, WideResNet
from utils.datasets import CIFAR10, FMNIST, MNIST, SequenceDataset #, CIFAR10Sub
from utils.litmodules import Classification
from utils.utils import ModelWithNormalization, dataloader, set_seed


def train(dataset_name: str, devices: List[int]) -> None:
    root = os.path.join('models', dataset_name)

    if os.path.exists(root):
        print(f'already exist: {root}')
        return
    else:
        os.makedirs(root)

    set_seed()

    dataset_root = os.path.join(os.path.sep, 'root', 'datasets')

    train_batch_size = 128
    n_class = 10

    if 'FMNIST' in dataset_name:
        dataset_cls = FMNIST
    elif 'MNIST' in dataset_name:
        dataset_cls = MNIST
    elif 'CIFAR10' in dataset_name:
        dataset_cls = CIFAR10
    else:
        raise ValueError(dataset_name)
    
    mean, std = dataset_cls.mean, dataset_cls.std

    if dataset_name in ('MNIST', 'FMNIST', 'CIFAR10'):
        train_dataset = dataset_cls(dataset_root, True)
        val_dataset = dataset_cls(dataset_root, False)

    else:
        if 'MNIST' in dataset_name: # including FMNIST
            transform = None
        elif 'CIFAR10' in dataset_name:
            transform = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()])
        else:
            raise ValueError(dataset_name)
        
        if 'MNIST_uniform' in dataset_name: # including FMNIST
            sd = dataset_name.split('_')
            ratio = float(sd[-1])
            dataset_name = '_'.join(sd[:-1])

        perturbation_dataset_path = os.path.join('datasets',  dataset_name, 'dataset')
        raw_dataset = torch.load(perturbation_dataset_path, map_location='cpu')
        imgs, labels = raw_dataset['imgs'], raw_dataset['labels']

        if 'MNIST_uniform' in dataset_name: # including FMNIST
            l = int(len(imgs) * ratio) # type: ignore
            imgs = imgs[:l]
            labels = labels[:l]

        #if 'CIFAR10_uniform_sub' in dataset_name:
        #    n_class = 2
        #    target_classes = [3, 9] if 'L2' in dataset_name else [0, 3]
        #    val_dataset = CIFAR10Sub(dataset_root, False, target_classes)
        #
        #    labels[labels == target_classes[0]] = 0
        #    labels[labels == target_classes[1]] = 1
        #
        #else:
        val_dataset = dataset_cls(dataset_root, False)

        train_dataset = SequenceDataset(imgs, labels, transform)

    train_dataloader = dataloader(train_dataset, train_batch_size, True, drop_last=True)
    val_dataloader = dataloader(val_dataset, len(val_dataset), False)
    
    if 'MNIST' in dataset_name: # including FMNIST
        classifier = ConvNet(n_class)
    elif 'CIFAR10' in dataset_name:
        classifier = WideResNet(28, 10, 0.3, n_class)
    else:
        raise ValueError(dataset_name)
    
    classifier = ModelWithNormalization(classifier, mean, std)

    optim = SGD
    optim_kwargs = {
        'lr': 0.1,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'nesterov': True,
    }

    # including FMNIST
    if 'MNIST_uniform' in dataset_name \
    or dataset_name in ('MNIST_natural_rand_L2', 'MNIST_natural_det_L2', 
                        'MNIST_natural_rand_Linf', 'MNIST_natural_det_Linf', 
                        'FMNIST_natural_rand_L2', 'FMNIST_natural_det_L2', 
                        'FMNIST_natural_rand_Linf', 'FMNIST_natural_det_Linf',
                        'CIFAR10_natural_rand_Linf'):
        optim_kwargs['lr'] = 0.01
    
    scheduler = ReduceLROnPlateau
    scheduler_kwargs = {}

    if 'MNIST_uniform' in dataset_name: # including FMNIST
        epochs = 300
    elif 'FMNIST' in dataset_name or 'CIFAR10' in dataset_name:
        epochs = 200
    elif 'MNIST' in dataset_name:
        epochs = 100
    else:
        raise ValueError(dataset_name)

    trainer = Trainer(
        logger=CSVLogger(root, name=None), # type: ignore
        enable_checkpointing=False,
        callbacks=EpochProgressBar(),
        default_root_dir=root, 
        devices=devices, 
        check_val_every_n_epoch=1,
        max_epochs=epochs,
        accelerator='gpu', 
        strategy='ddp_find_unused_parameters_false',
        precision=16,
        num_sanity_val_steps=0,
        deterministic=True,
    )

    litmodule = Classification(
        classifier, 
        n_class,
        optim, 
        optim_kwargs, 
        scheduler,
        scheduler_kwargs,
    )
        
    trainer.fit(litmodule, train_dataloader, val_dataloader) # type: ignore
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('devices', nargs='+', type=int)
    args = parser.parse_args()

    train(args.dataset_name, args.devices)