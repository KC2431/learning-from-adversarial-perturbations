from typing import Any, Callable, Optional, Tuple
from xml.etree import ElementTree as ET
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import os
from PIL import Image
import torch

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
            transform = T.Compose([T.Resize(256),
                                   T.RandomHorizontalFlip(),
                                   T.RandomVerticalFlip(),
                                   T.RandomRotation(degrees=45),
                                   T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                   T.CenterCrop(224),
                                   T.ToTensor(),
                                   ]
            )
            

        elif not train and transform is None:
            
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
        split = 'train' if train else 'val'
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform)

class FOOD101(torchvision.datasets.Food101):

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    n_class = 101
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

# Add normalization to the pipeline
            transform = T.Compose([
                    #T.Resize((232,232)),
                    #T.RandomHorizontalFlip(),
                    #T.RandomVerticalFlip(),
                    #T.RandomRotation(degrees=45),
                    #T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                    #T.CenterCrop(224),
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
            ])
            

        elif not train and transform is None:
            
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
        split = 'train' if train else 'test'
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform)

class IMAGENETTE(torchvision.datasets.Imagenette):

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    n_class = 10
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

# Add normalization to the pipeline
            transform = T.Compose([
                    T.RandomResizedCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
            ])
            

        elif not train and transform is None:
            
            transform = T.Compose([
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ])
        split = 'train' if train else 'val'
        super().__init__(root=root, split=split, transform=transform, target_transform=target_transform)

class BinaryDataset(Dataset):
    def __init__(self, original_dataset,which_dataset = 'imagenette'):
        self.dataset = original_dataset
        self.which_dataset = which_dataset
    def __getitem__(self, index):
        if self.which_dataset == 'imagenette':
            img, label = self.dataset[index]  # Get original image and label
        elif self.which_dataset == 'waterbirds':
            img, label, _ = self.dataset[index]  # Get original image and label
        if self.which_dataset == 'imagenette':
            new_label = -1 if label == 1 else 1  # Change label
        elif self.which_dataset == 'waterbirds':
            new_label = label.float()
        else:
            return NotImplementedError
        return img, new_label  # Return modified sample

    def __len__(self):
        return len(self.dataset)


class DatasetWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = self.transform(img)
        return img, label

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOCDataset(Dataset):
    def __init__(self, image_ids, root_dir, transform=None):
        self.image_ids = image_ids
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, "JPEGImages", f"{img_id}.jpg")
        xml_path = os.path.join(self.root_dir, "Annotations", f"{img_id}.xml")
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Load multi-label annotations
        labels = self._parse_voc_xml(xml_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels

    def _parse_voc_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = torch.zeros(20, dtype=torch.float32)
        
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name in VOC_CLASSES:
                labels[VOC_CLASSES.index(class_name)] = 1.0
        return labels

class VOCAugmentedDataset(Dataset):
    def __init__(self, image_ids, root_dir, transform=None):
        self.image_ids = image_ids
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_path = f"{self.root_dir}/JPEGImages/{img_id}.jpg"
        xml_path = f"{self.root_dir}/Annotations/{img_id}.xml"
        
        image = Image.open(img_path).convert("RGB")
        labels = self._parse_voc_xml(xml_path)  # Returns [20] tensor (multi-label)
        
        if self.transform:
            images = self.transform(image)  # List of 10 augmented tensors
            labels = labels.repeat(10, 1)   # Repeat labels for each augmentation
            return images, labels           # Shapes: [10,3,227,227], [10,20]
        return image, labels

    def _parse_voc_xml(self, xml_path):
        # Parse XML and return multi-hot vector [20]
        tree = ET.parse(xml_path)
        root = tree.getroot()
        labels = torch.zeros(20, dtype=torch.float32)
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name in VOC_CLASSES:
                labels[VOC_CLASSES.index(class_name)] = 1.0
        return labels

