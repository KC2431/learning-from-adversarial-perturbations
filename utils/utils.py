from typing import List, Literal, Union

import torch
from lightning_lite.utilities.seed import seed_everything
import json
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification.accuracy import MulticlassAccuracy
from torchvision.transforms.functional import normalize
from tqdm import tqdm


@torch.no_grad()
def in_range(x: Tensor, min: float, max: float) -> bool:
    return ((min<=x.min()) & (x.max()<=max)).item() # type: ignore


@torch.no_grad()
def all_elements_in_targets(x: Tensor, targets: List[float]) -> bool:
    return torch.isin(x, torch.tensor(targets, device=x.device)).all().item() # type: ignore


def freeze(model: Module) -> None:
    for p in model.parameters():
        p.requires_grad = False

class ExpLoss(Module):
    def __init__(self):
        super(ExpLoss, self).__init__()

    def forward(self, logits, labels):
        assert logits.shape == labels.shape
        assert len(logits.shape) == len(labels.shape) == 1
        return (- logits * labels).exp()
    

def get_model_device(model: Module) -> torch.device:
    return next(model.parameters()).device
        

def dataloader(
    dataset: Dataset, 
    batch_size: int, 
    shuffle: bool, 
    num_workers: int = 3, 
    pin_memory: bool = True, 
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def set_seed(seed: int = 0) -> None:
    seed_everything(seed, True)


def gpu(id: int) -> torch.device:
    print(torch.cuda.get_device_name(id))
    return torch.device(f'cuda:{id}') 


class ModelWithNormalization(Module):
    def __init__(self, model: Module, mean: List[float], std: List[float]) -> None:
        super().__init__()
        self.model = model
        self.mean = mean
        self.std = std

    def forward(self, x: Tensor) -> Tensor:
        assert in_range(x, 0, 1)
        return self.model(normalize(x, self.mean, self.std))

class CustomDataset(Dataset):
    def __init__(self, data_dict):
        self.imgs = data_dict["imgs"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]

class CalcClassificationAcc(LightningLite):
    def run(
        self, 
        classifier: Module, 
        loader: DataLoader, 
        n_class: int, 
        top_k: int = 1,
        average: Literal['micro', 'macro', 'weighted', 'none'] = 'micro',
    ) -> Union[float, List[float]]:

        classifier = self.setup(classifier)
        loader = self.setup_dataloaders(loader) # type: ignore

        freeze(classifier)
        classifier.eval()

        metric = MulticlassAccuracy(n_class, top_k, average)
        self.to_device(metric)

        for xs, labels in tqdm(loader):
            outs = classifier(xs)
            metric(outs, labels)
        
        acc = metric.compute()
        return acc.tolist() if average == 'none' else acc.item()
""" 
def load_config(config_path: str) -> Dict[str, Any]:
            config = json.load(f)
        print(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error parsing JSON configuration file at {config_path}")
        raise 
"""
