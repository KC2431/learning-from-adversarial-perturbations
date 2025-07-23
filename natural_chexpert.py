"""
import argparse
import os
import pathlib
from typing import Any, Dict, Literal

import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np

from utils.classifiers.binary_models import BinaryTrainedResNet
from utils.yang_datasets import CheXpertNoFinding
from utils.datasets import SequenceDataset
from utils.attacks import PGDL0, PGDL2, PGDLinf
from utils.utils import freeze, set_seed
from utils.gdpr_cfe import GDPR_CFE
from utils.scfe import APG0_CFE
from utils.utils import ModelWithNormalization

from uuid import uuid4

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
UUID = uuid4()

def calc_loss(outs: Tensor, labels: Tensor) -> Tensor:
    assert len(outs.shape) == 2
    assert len(labels.shape) == 1
    criterion = CrossEntropyLoss(reduction='none')
    return criterion(outs, labels)
 
class BinaryPGDL0(PGDL0):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDL2(PGDL2):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDLinf(PGDLinf):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

def fine_tune(classifier, dataloader, fname, method='Orig', train=False, save_model=True, seed=10, model_save_path='../SCRATCH/CFE_models') -> float: # type: ignore
    
    epochs = 10
    optim = SGD(classifier.parameters(), 
                lr=1e-3,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=False
            ) 
    
    scheduler = ReduceLROnPlateau(optim, factor=0.1, threshold=1e-7, verbose=True)
    classifier.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for _, (output) in enumerate(dataloader):
            if not train:
                imgs, labels = output[1], output[2]
            else: 
                imgs, labels = output[0], output[1]

            outs = classifier(imgs.to(classifier.device)) 
            losses = calc_loss(outs, labels.long().to(classifier.device))
            loss = losses.mean()

            optim.zero_grad(True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)

            optim.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step(running_loss/ len(dataloader.dataset))
        
        if epoch % 20 == 0:
            print(f'Running loss: {running_loss / len(dataloader.dataset):.2f}')
        if epoch == epochs - 1:
            if save_model:
                if method == 'Orig':
                    torch.save(classifier.state_dict(), f'{model_save_path}/cheXpert_trained_model_seed_{seed}.pt')
                else:
                    torch.save(classifier.state_dict(), f'{model_save_path}/{fname}.pt')
            return loss.item()


@torch.no_grad()
def test(classifier, dataloader) -> Tensor:
    num_samples = 0
    num_correct = 0
    
    for _, (_, imgs, labels, _) in enumerate(dataloader):
        assert len(labels.shape) == 1
        num_samples += len(labels)

        output = classifier(imgs.to(classifier.device))
        num_correct += (output.argmax(dim=1) == labels.view(-1).to(classifier.device)).count_nonzero().item()
    return num_correct / num_samples

@torch.no_grad()
def get_attack_succ_rate(classifier, dataloader):
    num_succ_attacks = 0
    for _, (adv_img, adv_labels) in enumerate(dataloader):
        pred = classifier(adv_img.to(classifier.device))
        assert len(pred.shape) == 2
        assert len(adv_labels.shape) == 1
        num_succ_attacks += (pred.argmax(dim=1) == adv_labels.to(classifier.device)).count_nonzero().item()

    return num_succ_attacks / len(dataloader.dataset)

def to_cpu(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = v.cpu()
        elif isinstance(v, torch.nn.Module):
            d[k] = v.cpu().state_dict()
    return d

def generate_adv_labels(n: int, device: torch.device) -> Tensor:
    return torch.randint(0, 2, (n,), device=device)

class Main(LightningLite):
    def run(self,
        norm: Literal['GDPR_CFE', 'SCFE', 'L2', 'Linf'],
        seed: int,
    ) -> None:

        print(f'UUID: {UUID}')

        models_path = '../SCRATCH/CFE_models'
        dataset_path = '../SCRATCH'

        root = '/home/htc/kchitranshi/SCRATCH/CFE_datasets'
        os.makedirs(root, exist_ok=True)

        fname = f'cheXpert_{norm}_seed_{seed}'
        path = os.path.join(root, fname)

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            pathlib.Path(path).touch()

        # Fine tuning the model and setting the seed
        fine_tune_model = True
        set_seed(seed)

        # Defining transformations
        train_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])

        adv_train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
        ])

        val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])

        # Loading the datasets
        train_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='tr',hparams=None)
        val_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='va',hparams=None)
        test_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='te',hparams=None)
        train_test_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='tr',hparams=None)

        # If fine tune model
        if fine_tune_model:
            train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                        batch_size=108, 
                                                        shuffle=True,
                                                        num_workers=3,
                                                        pin_memory=True
                        )

        # Loading validation and test datasets
        val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                        batch_size=256, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        test_dataloader = torch.utils.data.DataLoader(test_data, 
                                                        batch_size=256, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        train_test_dataloader = torch.utils.data.DataLoader(train_test_data, 
                                                        batch_size=256, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        # Loading the model
        model = BinaryTrainedResNet('resnet50', num_classes=2)
        if fine_tune_model:
            print("Training a model on clean dataset.")
            model.train()
        else:
            print(f"Loading Pre-trained Model.")
            state_dict = torch.load(f"{models_path}/cheXpert_trained_model_seed_{seed}.pt", map_location='cpu')
        classifier = ModelWithNormalization(model,
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]
                    )
        if not fine_tune_model:
            classifier.load_state_dict(state_dict)

        # Fine Tuning the classifier
        classifier = self.setup(classifier)
        loss = fine_tune(classifier, 
                         train_dataloader, 
                         fname=fname, 
                         method='Orig', 
                         save_model=True, 
                         seed=seed, 
                         model_save_path=models_path) if fine_tune_model else None
        freeze(classifier)
        classifier.eval()

        # Calculating accuracy on validation and test sets
        test_acc = test(classifier, test_dataloader)
        val_acc = test(classifier, val_dataloader)
        train_test_acc = test(classifier, train_test_dataloader)

        print('-'*60)
        print(f'Accuracy of trained model on clean train data: {train_test_acc * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of trained model on clean val data: {val_acc * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of trained model on clean test data: {test_acc * 100:.2f}%')
        print('-'*60)

        data_range = (0,1)
        steps = 100

        # Defining the adversarial attacks and CounterFactual Explanations
        if norm == 'GDPR_CFE':
            atk = GDPR_CFE(
                    model=classifier,
                    max_image_range = 1.0,
                    min_image_range = 0.0, 
                    optimizer = torch.optim.Adam, 
                    iters=steps, 
                    lamb=1e-2,
                    lamb_cf=1e-2,
                    mode="natural_binary",
                    device= classifier.device,
                )
        elif norm == 'SCFE':
            atk = APG0_CFE
        elif norm == 'L2':
            atk = BinaryPGDL2(classifier=classifier, 
                              steps=steps, 
                              eps=3, 
                              data_range=data_range
                )
        elif norm == 'Linf':
            atk = BinaryPGDLinf(classifier=classifier, 
                                steps=steps, 
                                eps=0.03, 
                                data_range=data_range
                )
        else:
            raise ValueError(norm)

        del train_data 
        adv_dataset = {'imgs': [], 'labels': []}

        # Intialising the data to be attacked
        adv_attack_data = CheXpertNoFinding(data_path=f'{dataset_path}',split='tr',hparams=None)
        adv_attack_loader = torch.utils.data.DataLoader(adv_attack_data, 
                                                        batch_size=256, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                            )

        # Conducting the adversarial attacks/CFEs
        print("Conducting Adversarial Attacks/CFEs.")
        avg_L2_norms = []
        for _, (_, data, _, _) in tqdm(enumerate(adv_attack_loader)):

            # Generating target labels            
            labels = generate_adv_labels(data.shape[0], classifier.device)

            if norm in ['L2','Linf']:
                adv_data = atk(data, labels)
            elif norm == 'SCFE':
                data_flattened = data.view(data.size(0), -1)

                maxs = torch.tensor(data_range[1]).repeat(data_flattened.size(-1))
                mins = torch.tensor(data_range[0]).repeat(data_flattened.size(-1))

                del data_flattened

                cfe_atk = atk(model=classifier,
                           range_min=None,
                           range_max=None,
                           numclasses=2,
                           scale_model=False,
                           iters=steps,
                           maxs=maxs,
                           mins=mins,
                           lam_steps=3,
                           L0=1e-5, # changed from 1e-4 to 1e-3
                           num_col_channels=data.shape[1],
                           height=data.shape[2],
                           width=data.shape[3]
                )
                adv_data = cfe_atk.get_CFs_natural_binary(data.to(classifier.device), labels.unsqueeze(1).to(classifier.device))

                del maxs
                del mins
            else:
                adv_data = atk.get_perturbations(data, labels.unsqueeze(1))
                
            avg_L2_norms.append((adv_data.cpu() - data).norm(p=2, dim=(1,2,3)).mean().item())
            adv_dataset['imgs'].append(adv_data.cpu())
            adv_dataset['labels'].append(labels.cpu())
        
        # Concating the adversarial/CFE data
        adv_dataset['imgs'] = torch.cat(adv_dataset['imgs'])
        adv_dataset['labels'] = torch.cat(adv_dataset['labels'])

        # Defining the dataloaders for the adversarial/CFE data
        adv_data = SequenceDataset(adv_dataset['imgs'], adv_dataset['labels'],x_transform=None)
        adv_asr_check_data = SequenceDataset(adv_dataset['imgs'], adv_dataset['labels'],x_transform=None)

        del adv_dataset

        adv_dataloader = torch.utils.data.DataLoader(adv_data, 
                                                        batch_size=108, 
                                                        shuffle=True,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        adv_asr_check_dataloader = torch.utils.data.DataLoader(adv_asr_check_data, 
                                                        batch_size=128, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        
        # Checking the Attack Success Rate for Adversarial attacks/CFEs
        print('-'*60)
        attack_succ_rate = get_attack_succ_rate(classifier=classifier, dataloader=adv_asr_check_dataloader)
        print(f'The attack success rate is {attack_succ_rate * 100:.2f} with an avg L2 norm of {np.mean(avg_L2_norms).item():.2f}')
        print('-'*60)

        print("Training a model on perturbed data.")
        # Intialising the Adversarial model
        adv_model = BinaryTrainedResNet(resnet_type='resnet50', num_classes=2)
        adv_model.train()
        adv_classifier = ModelWithNormalization(adv_model,
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]
                    )
        adv_classifier = self.setup(adv_classifier)
        adv_loss = fine_tune(adv_classifier, 
                             adv_dataloader, 
                             fname=fname, 
                             method=norm, 
                             save_model=True, 
                             model_save_path=models_path)
        freeze(adv_classifier)
        adv_classifier.eval()

        # Checking its accuracy on validation/test sets
        adv_acc_for_natural_train_test = test(adv_classifier, train_test_dataloader)
        adv_acc_for_natural_val = test(adv_classifier, val_dataloader)
        adv_acc_for_natural_test = test(adv_classifier, test_dataloader)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean train data: {adv_acc_for_natural_train_test * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean val data: {adv_acc_for_natural_val * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean test data: {adv_acc_for_natural_test * 100:.2f}%')
        print('-'*60)

        print('='*60)
        print('='*60)

        save_data = {
            'classifier': classifier,
            'adv_data': adv_data,
            'clean_fine_tune_loss': loss if fine_tune_model else None,
            'adv_classifier': adv_classifier,
            'adv_train_loss': adv_loss,
            'adv_acc_for_natural_test': adv_acc_for_natural_test,
            'avg_L2_norms': avg_L2_norms
        }
        to_cpu(save_data)
        torch.save(save_data, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('norm', choices=('GDPR_CFE', 'SCFE', 'L2', 'Linf'))
    parser.add_argument('seed', type=int)
    parser.add_argument('devices', nargs='+', type=int)
    args = parser.parse_args()

    lite_kwargs = {
        'accelerator': 'gpu',
        'strategy': 'ddp_find_unused_parameters_false',
        'devices': args.devices,
        'precision': 16,
    }
    
    Main(**lite_kwargs).run(
        args.norm,
        args.seed,
    )
"""
import argparse
import json
import os
import pathlib
from typing import Any, Dict, Literal

import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np

from utils.classifiers.binary_models import BinaryTrainedResNet
from utils.yang_datasets import CheXpertNoFinding
from utils.datasets import SequenceDataset
from utils.attacks import PGDL0, PGDL2, PGDLinf
from utils.utils import freeze, set_seed
from utils.gdpr_cfe import GDPR_CFE
from utils.scfe import APG0_CFE
from utils.utils import ModelWithNormalization

from uuid import uuid4

UUID = uuid4()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_environment(config: Dict[str, Any]) -> None:
    """Setup environment variables."""
    env_config = config.get("environment", {})
    if "cublas_workspace_config" in env_config:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = env_config["cublas_workspace_config"]

def calc_loss(outs: Tensor, labels: Tensor) -> Tensor:
    assert len(outs.shape) == 2
    assert len(labels.shape) == 1
    criterion = CrossEntropyLoss(reduction='none')
    return criterion(outs, labels)
 
class BinaryPGDL0(PGDL0):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDL2(PGDL2):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDLinf(PGDLinf):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

def get_optimizer(model_params, config: Dict[str, Any], optimizer_key: str = "optimizer"):
    """Create optimizer based on configuration."""
    opt_config = config[optimizer_key]
    opt_type = opt_config["type"]
    
    if opt_type == "SGD":
        return SGD(
            model_params,
            lr=opt_config["lr"],
            momentum=opt_config["momentum"],
            weight_decay=opt_config["weight_decay"],
            nesterov=opt_config["nesterov"]
        )
    elif opt_type == "Adam":
        return Adam(
            model_params,
            lr=opt_config["lr"],
            weight_decay=opt_config.get("weight_decay", 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

def get_scheduler(optimizer, config: Dict[str, Any]):
    """Create scheduler based on configuration."""
    sched_config = config.get("scheduler", {})
    sched_type = sched_config.get("type", "ReduceLROnPlateau")
    
    if sched_type == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer, 
            factor=sched_config.get("factor", 0.1), 
            threshold=sched_config.get("threshold", 1e-7), 
            verbose=sched_config.get("verbose", True)
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {sched_type}")

def fine_tune(classifier, dataloader, fname, config: Dict[str, Any], method='Orig', train=False, save_model=True, seed=10) -> float:
    training_config = config["training"]
    reg_config = config["regularization"]
    paths_config = config["paths"]
    eval_config = config["evaluation"]
    
    epochs = training_config["epochs"]
    optim = get_optimizer(classifier.parameters(), config)
    scheduler = get_scheduler(optim, config)
    
    classifier.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for _, (output) in enumerate(dataloader):
            if not train:
                imgs, labels = output[1], output[2]
            else: 
                imgs, labels = output[0], output[1]

            outs = classifier(imgs.to(classifier.device)) 
            losses = calc_loss(outs, labels.long().to(classifier.device))
            loss = losses.mean()

            optim.zero_grad(True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(classifier.parameters(), reg_config["grad_clip_norm"])

            optim.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step(running_loss / len(dataloader.dataset))
        
        print_interval = eval_config.get("print_interval", 20)
        if epoch % print_interval == 0:
            print(f'Running loss: {running_loss / len(dataloader.dataset):.2f}')
        if epoch == epochs - 1:
            if save_model:
                models_path = paths_config["models_path"]
                if method == 'Orig':
                    torch.save(classifier.state_dict(), f'{models_path}/cheXpert_trained_model_seed_{seed}.pt')
                else:
                    torch.save(classifier.state_dict(), f'{models_path}/{fname}.pt')
            return loss.item()


@torch.no_grad()
def test(classifier, dataloader) -> Tensor:
    num_samples = 0
    num_correct = 0
    
    for _, (_, imgs, labels, _) in enumerate(dataloader):
        assert len(labels.shape) == 1
        num_samples += len(labels)

        output = classifier(imgs.to(classifier.device))
        num_correct += (output.argmax(dim=1) == labels.view(-1).to(classifier.device)).count_nonzero().item()
    return num_correct / num_samples

@torch.no_grad()
def get_attack_succ_rate(classifier, dataloader):
    num_succ_attacks = 0
    for _, (adv_img, adv_labels) in enumerate(dataloader):
        pred = classifier(adv_img.to(classifier.device))
        assert len(pred.shape) == 2
        assert len(adv_labels.shape) == 1
        num_succ_attacks += (pred.argmax(dim=1) == adv_labels.to(classifier.device)).count_nonzero().item()

    return num_succ_attacks / len(dataloader.dataset)

def to_cpu(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = v.cpu()
        elif isinstance(v, torch.nn.Module):
            d[k] = v.cpu().state_dict()
    return d

def generate_adv_labels(n: int, device: torch.device) -> Tensor:
    return torch.randint(0, 2, (n,), device=device)

def create_transforms(config: Dict[str, Any]):
    """Create data transforms based on configuration."""
    data_config = config["data"]
    transforms_config = data_config["transforms"]
    
    train_transform = T.Compose([
        T.Resize(transforms_config["train"]["resize"]),
        T.CenterCrop(transforms_config["train"]["center_crop"]),
        T.ToTensor()
    ])

    adv_train_transform = T.Compose([
        T.RandomResizedCrop(transforms_config["adv_train"]["random_resized_crop"]),
        T.RandomHorizontalFlip(),
    ])

    val_transform = T.Compose([
        T.Resize(transforms_config["val"]["resize"]),
        T.CenterCrop(transforms_config["val"]["center_crop"]),
        T.ToTensor()
    ])
    
    return train_transform, adv_train_transform, val_transform

def create_attack(norm: str, classifier, config: Dict[str, Any]):
    """Create attack based on norm type and configuration."""
    attack_config = config["adversarial_attacks"]
    
    if norm == 'GDPR_CFE':
        gdpr_config = attack_config["GDPR_CFE"]
        return GDPR_CFE(
            model=classifier,
            max_image_range=gdpr_config["max_image_range"],
            min_image_range=gdpr_config["min_image_range"], 
            optimizer=getattr(torch.optim, gdpr_config["optimizer"]), 
            iters=gdpr_config["iters"], 
            lamb=gdpr_config["lamb"],
            lamb_cf=gdpr_config["lamb_cf"],
            mode=gdpr_config["mode"],
            device=classifier.device,
        )
    elif norm == 'SCFE':
        return APG0_CFE
    elif norm == 'L2':
        l2_config = attack_config["L2"]
        return BinaryPGDL2(
            classifier=classifier, 
            steps=l2_config["steps"], 
            eps=l2_config["eps"], 
            data_range=l2_config["data_range"]
        )
    elif norm == 'Linf':
        linf_config = attack_config["Linf"]
        return BinaryPGDLinf(
            classifier=classifier, 
            steps=linf_config["steps"], 
            eps=linf_config["eps"], 
            data_range=linf_config["data_range"]
        )
    else:
        raise ValueError(f"Unsupported norm: {norm}")

class Main(LightningLite):
    def run(self,
        norm: Literal['GDPR_CFE', 'SCFE', 'L2', 'Linf'],
        seed: int,
        config: Dict[str, Any]
    ) -> None:

        print(f'UUID: {UUID}')

        # Extract configuration sections
        training_config = config["training"]
        paths_config = config["paths"]
        data_config = config["data"]
        model_config = config["model"]
        attack_config = config["adversarial_attacks"]
        dataset_config = config["dataset"]

        models_path = paths_config["models_path"]
        dataset_path = paths_config["dataset_path"]
        root = paths_config["root"]

        os.makedirs(root, exist_ok=True)

        fname = f'cheXpert_{norm}_seed_{seed}'
        path = os.path.join(root, fname)

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            pathlib.Path(path).touch()

        # Fine tuning the model and setting the seed
        fine_tune_model = training_config["fine_tune_model"]
        set_seed(seed)

        # Create transforms
        train_transform, adv_train_transform, val_transform = create_transforms(config)

        # Loading the datasets
        dataset_name = dataset_config["name"]
        splits = dataset_config["splits"]
        
        train_data = CheXpertNoFinding(data_path=dataset_path, split=splits["train"], hparams=None)
        val_data = CheXpertNoFinding(data_path=dataset_path, split=splits["validation"], hparams=None)
        test_data = CheXpertNoFinding(data_path=dataset_path, split=splits["test"], hparams=None)
        train_test_data = CheXpertNoFinding(data_path=dataset_path, split=splits["train"], hparams=None)

        # Create dataloaders with config parameters
        batch_size = training_config["batch_size"]
        test_batch_size = training_config["test_batch_size"]
        adv_batch_size = training_config["adv_batch_size"]
        num_workers = training_config["num_workers"]
        pin_memory = training_config["pin_memory"]

        # If fine tune model
        if fine_tune_model:
            train_dataloader = torch.utils.data.DataLoader(
                train_data, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory
            )

        # Loading validation and test datasets
        val_dataloader = torch.utils.data.DataLoader(
            val_data, 
            batch_size=test_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=test_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        train_test_dataloader = torch.utils.data.DataLoader(
            train_test_data, 
            batch_size=test_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Loading the model
        model = BinaryTrainedResNet(model_config["architecture"], num_classes=model_config["num_classes"])
        if fine_tune_model:
            print("Training a model on clean dataset.")
            model.train()
        else:
            print(f"Loading Pre-trained Model.")
            state_dict = torch.load(f"{models_path}/cheXpert_trained_model_seed_{seed}.pt", map_location='cpu')
        
        norm_config = model_config["normalization"]
        classifier = ModelWithNormalization(
            model,
            mean=norm_config["mean"], 
            std=norm_config["std"]
        )
        if not fine_tune_model:
            classifier.load_state_dict(state_dict)

        # Fine Tuning the classifier
        classifier = self.setup(classifier)
        loss = fine_tune(
            classifier, 
            train_dataloader, 
            fname=fname, 
            config=config,
            method='Orig', 
            save_model=True, 
            seed=seed
        ) if fine_tune_model else None
        freeze(classifier)
        classifier.eval()

        # Calculating accuracy on validation and test sets
        test_acc = test(classifier, test_dataloader)
        val_acc = test(classifier, val_dataloader)
        train_test_acc = test(classifier, train_test_dataloader)

        print('-'*60)
        print(f'Accuracy of trained model on clean train data: {train_test_acc * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of trained model on clean val data: {val_acc * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of trained model on clean test data: {test_acc * 100:.2f}%')
        print('-'*60)

        data_range = tuple(attack_config["data_range"])
        steps = attack_config["steps"]

        # Create attack
        atk = create_attack(norm, classifier, config)

        del train_data 
        adv_dataset = {'imgs': [], 'labels': []}

        # Intialising the data to be attacked
        adv_attack_data = CheXpertNoFinding(data_path=dataset_path, split=splits["train"], hparams=None)
        adv_attack_loader = torch.utils.data.DataLoader(
            adv_attack_data, 
            batch_size=test_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        # Conducting the adversarial attacks/CFEs
        print("Conducting Adversarial Attacks/CFEs.")
        avg_L2_norms = []
        for _, (_, data, _, _) in tqdm(enumerate(adv_attack_loader)):

            # Generating target labels            
            labels = generate_adv_labels(data.shape[0], classifier.device)

            if norm in ['L2','Linf']:
                adv_data = atk(data, labels)
            elif norm == 'SCFE':
                scfe_config = attack_config["SCFE"]
                data_flattened = data.view(data.size(0), -1)

                maxs = torch.tensor(data_range[1]).repeat(data_flattened.size(-1))
                mins = torch.tensor(data_range[0]).repeat(data_flattened.size(-1))

                del data_flattened

                cfe_atk = atk(
                    model=classifier,
                    range_min=None,
                    range_max=None,
                    numclasses=model_config["num_classes"],
                    scale_model=scfe_config["scale_model"],
                    iters=scfe_config["iters"],
                    maxs=maxs,
                    mins=mins,
                    lam_steps=scfe_config["lam_steps"],
                    L0=scfe_config["L0"],
                    num_col_channels=data.shape[1],
                    height=data.shape[2],
                    width=data.shape[3]
                )
                adv_data = cfe_atk.get_CFs_natural_binary(data.to(classifier.device), labels.unsqueeze(1).to(classifier.device))

                del maxs
                del mins
            else:
                adv_data = atk.get_perturbations(data, labels.unsqueeze(1))
                
            avg_L2_norms.append((adv_data.cpu() - data).norm(p=2, dim=(1,2,3)).mean().item())
            adv_dataset['imgs'].append(adv_data.cpu())
            adv_dataset['labels'].append(labels.cpu())
        
        # Concating the adversarial/CFE data
        adv_dataset['imgs'] = torch.cat(adv_dataset['imgs'])
        adv_dataset['labels'] = torch.cat(adv_dataset['labels'])

        # Defining the dataloaders for the adversarial/CFE data
        adv_data = SequenceDataset(adv_dataset['imgs'], adv_dataset['labels'], x_transform=None)
        adv_asr_check_data = SequenceDataset(adv_dataset['imgs'], adv_dataset['labels'], x_transform=None)

        del adv_dataset

        adv_dataloader = torch.utils.data.DataLoader(
            adv_data, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        adv_asr_check_dataloader = torch.utils.data.DataLoader(
            adv_asr_check_data, 
            batch_size=adv_batch_size, 
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        # Checking the Attack Success Rate for Adversarial attacks/CFEs
        print('-'*60)
        attack_succ_rate = get_attack_succ_rate(classifier=classifier, dataloader=adv_asr_check_dataloader)
        print(f'The attack success rate is {attack_succ_rate * 100:.2f} with an avg L2 norm of {np.mean(avg_L2_norms).item():.2f}')
        print('-'*60)

        print("Training a model on perturbed data.")
        # Intialising the Adversarial model
        adv_model = BinaryTrainedResNet(resnet_type=model_config["architecture"], num_classes=model_config["num_classes"])
        adv_model.train()
        adv_classifier = ModelWithNormalization(
            adv_model,
            mean=norm_config["mean"], 
            std=norm_config["std"]
        )
        adv_classifier = self.setup(adv_classifier)
        adv_loss = fine_tune(
            adv_classifier, 
            adv_dataloader, 
            fname=fname, 
            config=config,
            method=norm, 
            save_model=True
        )
        freeze(adv_classifier)
        adv_classifier.eval()

        # Checking its accuracy on validation/test sets
        adv_acc_for_natural_train_test = test(adv_classifier, train_test_dataloader)
        adv_acc_for_natural_val = test(adv_classifier, val_dataloader)
        adv_acc_for_natural_test = test(adv_classifier, test_dataloader)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean train data: {adv_acc_for_natural_train_test * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean val data: {adv_acc_for_natural_val * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean test data: {adv_acc_for_natural_test * 100:.2f}%')
        print('-'*60)

        print('='*60)
        print('='*60)

        save_data = {
            'classifier': classifier,
            'adv_data': adv_data,
            'clean_fine_tune_loss': loss if fine_tune_model else None,
            'adv_classifier': adv_classifier,
            'adv_train_loss': adv_loss,
            'adv_acc_for_natural_test': adv_acc_for_natural_test,
            'avg_L2_norms': avg_L2_norms
        }
        to_cpu(save_data)
        torch.save(save_data, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('norm', choices=('GDPR_CFE', 'SCFE', 'L2', 'Linf'))
    parser.add_argument('seed', type=int)
    parser.add_argument('devices', nargs='+', type=int)
    parser.add_argument('config', type=str, help='Path to configuration JSON file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup environment
    setup_environment(config)
    
    # Extract Lightning configuration
    lightning_config = config["lightning"]
    lite_kwargs = {
        'accelerator': lightning_config["accelerator"],
        'strategy': lightning_config["strategy"],
        'devices': args.devices,
        'precision': lightning_config["precision"],
    }
    
    Main(**lite_kwargs).run(
        args.norm,
        args.seed,
        config,
    )