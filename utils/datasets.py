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
import pandas as pd

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

        assert self.which_dataset in ['imagenette', 'waterbirds', 'spuco_dogs'], 'Dataset not supported'
    def __getitem__(self, index):
        if self.which_dataset == 'imagenette':
            img, label = self.dataset[index]  # Get original image and label
        elif self.which_dataset in ['waterbirds','spuco_dogs']:
            img, label, _ = self.dataset[index]  # Get original image and label
        if self.which_dataset == 'imagenette':
            new_label = -1 if label == 1 else 1  # Change label
        elif self.which_dataset in ['waterbirds', 'spuco_dogs']:
            new_label = label
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
    
class SpuCoDogsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        self.contexts = []
        self.classes = []
        self.context_to_idx = {}
        self.class_to_idx = {}

        for cls in sorted(os.listdir(root)):
            class_path = os.path.join(root, cls)
            if not os.path.isdir(class_path): continue

            for ctx in sorted(os.listdir(class_path)):
                context_path = os.path.join(class_path, ctx)
                if not os.path.isdir(context_path): continue

                self.class_to_idx.setdefault(cls, len(self.class_to_idx))
                self.context_to_idx.setdefault(ctx, len(self.context_to_idx))

                for fname in os.listdir(context_path):
                    if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                        img_path = os.path.join(context_path, fname)
                        self.samples.append((img_path, cls, ctx))

        self.classes = list(self.class_to_idx.keys())
        self.contexts = list(self.context_to_idx.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cls, ctx = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[cls]
        context = self.context_to_idx[ctx]
        return image, label, context

class CheXpertBinaryDataset(Dataset):
    """
    CheXpert Binary Classification Dataset
    
    This dataset specifically focuses on the binary classification task of 
    determining whether a chest X-ray shows "No Finding" (normal) or has pathology.
    """
    
    def __init__(self, csv_path, image_root_path, transform=None, uncertain_policy='zero'):
        """
        Args:
            csv_path (string): Path to the CSV file with annotations
            image_root_path (string): Root directory of the images
            transform (callable, optional): Optional transform to be applied on a sample
            uncertain_policy (string): How to handle uncertain labels (U)
                                      Options: 'zero', 'one', 'ignore'
        """
        self.data_frame = pd.read_csv(csv_path)
        self.image_root_path = image_root_path
        self.transform = transform
        self.uncertain_policy = uncertain_policy
        
        # For binary classification, we focus on "No Finding" column
        # Make sure it exists in the dataset
        if 'No Finding' not in self.data_frame.columns:
            raise ValueError("'No Finding' column not found in the CSV file")
        
        # Process uncertain labels for "No Finding"
        self._process_no_finding_label()
        
    def _process_no_finding_label(self):
        """Process 'No Finding' label according to the uncertain policy"""
        if self.uncertain_policy == 'zero':
            # Convert uncertain (-1) to 0
            self.data_frame['No Finding'] = self.data_frame['No Finding'].replace(-1.0, 0.0)
        elif self.uncertain_policy == 'one':
            # Convert uncertain (-1) to 1
            self.data_frame['No Finding'] = self.data_frame['No Finding'].replace(-1.0, 1.0)
        # For 'ignore', we leave -1 values as is
        
        # Convert NaN to 0 (which means there is some finding)
        self.data_frame['No Finding'] = self.data_frame['No Finding'].fillna(0.0)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path - typically the 'Path' column in CheXpert
        img_path = os.path.join(self.image_root_path, self.data_frame.iloc[idx]['Path'])
        
        # Read image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            
        # Get binary label for "No Finding"
        label = self.data_frame.iloc[idx]['No Finding']
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        return image, label