from typing import Any, Callable, Optional, Tuple

import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self, 
        xs: Any, 
        ys: Any, 
        x_transform: Optional[Callable] = None, 
        y_transform: Optional[Callable] = None,
    ) -> None:
        assert len(xs) == len(ys)
        self.xs = xs
        self.ys = ys
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self) -> int:
        return len(self.xs)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x, y = self.xs[idx], self.ys[idx]
        if self.x_transform:
            x = self.x_transform(x)
        if self.y_transform:
            y = self.y_transform(y)
        return x, y
    

# `mean` and `std` must be list (not tuple)
# to be compatible with the type hint of `torchvision.transforms.functional.normalize`.
    

class MNIST(torchvision.datasets.MNIST):
    mean = [0.1307]
    std = [0.3081]
    n_class = 10
    classes = tuple(range(10))
    size = (1, 28, 28)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        transform = T.ToTensor() if transform is None else transform
        super().__init__(root, train, transform, target_transform, True)


class FMNIST(torchvision.datasets.FashionMNIST):
    mean = [0.2860]
    std = [0.3530]
    n_class = 10
    #classes = ('T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
    #           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')
    classes = ('Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')
    size = (1, 28, 28)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        transform = T.ToTensor() if transform is None else transform
        super().__init__(root, train, transform, target_transform, True)


class CIFAR10(torchvision.datasets.CIFAR10):
    # This must be list (not tuple)
    # to be compatible with the type hint of `torchvision.transforms.functional.normalize`.
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    n_class = 10
    classes = ('Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck')
    size = (3, 32, 32)
    dim = size[0] * size[1] * size[2]

    def __init__(
        self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:

        if train and transform is None:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

        elif not train and transform is None:
            transform = T.ToTensor()

        super().__init__(root, train, transform, target_transform, True)

class IMAGENET(torchvision.datasets.ImageNet):

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    n_class = 1000
    size=(3, 224, 224)
    dim = size[0] * size[1] * size[2]

    def __init__(self, 
        root: str, 
        train: bool, 
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None
    ) -> None:
        pass

        if train and transform is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Add normalization to the pipeline
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                normalize,
            ])

        elif not train and transform is None:
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
        split = 'train' if train else 'val'
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform)
        
'''
from typing import List

import torch


class CIFAR10Sub(CIFAR10):
    def __init__(
        self, 
        root: str, 
        train: bool, 
        use_classes: List[int],
        transform: Optional[Callable] = None, 
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, train, transform, target_transform)

        self.n_class = len(use_classes)

        self.targets = torch.tensor(self.targets)
        indices = torch.zeros(len(self.targets), dtype=bool)
        for c in use_classes:
            indices.logical_or_(self.targets == c)
            
        self.data = self.data[indices]
        self.targets = self.targets[indices]

        for i, c in enumerate(use_classes):
            self.targets[self.targets == c] = i
'''
