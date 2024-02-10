import os
from typing import Any, Dict

import torch
import yaml
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, ProgressBarBase
from torch.utils.data import DataLoader
from tqdm import tqdm


def _get_class_path(obj: Any) -> str:
    return str(obj.__class__).lstrip("<class '").rstrip("'>")


def _dump(path: str, dic: Dict[str, Any]) -> None:
    with open(path, 'w') as f:
        yaml.dump(dic, f)


class SaveSettings(Callback):
    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        root = self._get_root(trainer)
        path = os.path.join(root, 'fit_settings.yaml')

        d = self._shared_step(trainer, pl_module)
        d.update({
            'num_sanity_val_steps': trainer.num_sanity_val_steps,
            'optimizers': str(trainer.optimizers),
            'schedulers': str(trainer.lr_scheduler_configs),
            'max_epochs': trainer.max_epochs,
            'max_steps': trainer.max_steps,
            'train_dataloader': self._extract_dataloader_settings(trainer.train_dataloader.loaders), # type: ignore
        })

        if trainer.val_dataloaders is not None:
            for i, loader in enumerate(trainer.val_dataloaders):
                d.update({f'val_dataloader_{i}': self._extract_dataloader_settings(loader)})

        _dump(path, d)


    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        root = self._get_root(trainer)
        path = os.path.join(root, 'test_settings.yaml')

        d = self._shared_step(trainer, pl_module)
        for i, loader in enumerate(trainer.test_dataloaders): # type: ignore
            d.update({f'test_dataloader_{i}': self._extract_dataloader_settings(loader)})
            
        _dump(path, d)


    @staticmethod
    def _get_root(trainer: Trainer) -> str:
        return os.path.join(trainer.log_dir, f'version_{trainer.logger.version}') # type: ignore


    @staticmethod
    def _shared_step(trainer: Trainer, pl_module: LightningModule) -> Dict[str, Any]:
        return {
            'seed': os.environ.get('PL_GLOBAL_SEED', None),
            'device': torch.cuda.get_device_name(),
            'accelerator': _get_class_path(trainer.accelerator),
            'strategy': _get_class_path(trainer.strategy),
            'amp_backend': str(trainer.amp_backend),
            'precision': trainer.precision,
            'gpu_numbers': trainer.num_devices,
        }


    @staticmethod
    def _extract_dataloader_settings(dataloader: DataLoader) -> Dict[str, Any]:
        return {
            'batch_size': dataloader.batch_size,
            'num_workers': dataloader.num_workers,
            'pin_memory': dataloader.pin_memory,
            'drop_last': dataloader.drop_last,
            'sampler': _get_class_path(dataloader.sampler),
            'shuffle': getattr(dataloader.sampler, 'shuffle', None),
        }


class EpochProgressBar(ProgressBarBase):
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.bar = tqdm(desc='Epoch', leave=False, total=trainer.max_epochs)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.bar.update(1)


class LearningRateLogger(Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module.log('lr', trainer.optimizers[0].param_groups[0]['lr'],
                      on_step=False, on_epoch=True, sync_dist=True)