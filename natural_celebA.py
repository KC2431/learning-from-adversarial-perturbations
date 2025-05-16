import argparse
from collections import OrderedDict
import os
import pathlib
from typing import Any, Dict, Literal

import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.datasets import CelebA
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np

from utils.classifiers.binary_models import BinaryTrainedResNet
from utils.datasets import SequenceDataset
from utils.attacks import PGDL0, PGDL2, PGDLinf
from utils.utils import freeze, set_seed
from utils.gdpr_cfe import GDPR_CFE
from utils.scfe import APG0_CFE
from utils.utils import ModelWithNormalization

from wilds import get_dataset
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
    

def fine_tune(classifier, dataloader, train=False, save_model=True) -> float: # type: ignore
    
    epochs = 20
    optim = SGD(classifier.parameters(), 
                lr=1e-3,
                momentum=0.9,
                weight_decay=1e-4,
                nesterov=False
            ) 
    
    classifier.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for _, (imgs, labels) in enumerate(dataloader):
            outs = classifier(imgs.cuda())
            labels = labels[:,9] if not train else labels 
            losses = calc_loss(outs, labels.cuda())
            loss = losses.mean()

            optim.zero_grad(True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)

            optim.step()
            running_loss += loss.item() * imgs.size(0)

        if epoch % 20 == 0:
            print(f'Running loss: {running_loss / len(dataloader.dataset):.2f}')
        if epoch == epochs - 1:
            if save_model:
                torch.save(classifier.state_dict(), f'../SCRATCH/CFE_models/celebA_{UUID}.pt')
            return loss.item()


@torch.no_grad()
def test(classifier, dataloader) -> Tensor:
    num_samples = 0
    num_correct = 0
    
    for _, (imgs, labels) in enumerate(dataloader):
        labels = labels[:,9] 
        assert len(labels.shape) == 1
        num_samples += len(labels)

        output = classifier(imgs.cuda())
        num_correct += (output.argmax(dim=1) == labels.cuda()).count_nonzero().item()
    return num_correct / num_samples

@torch.no_grad()
def get_attack_succ_rate(classifier, dataloader):
    num_succ_attacks = 0
    for _, (adv_img, adv_labels) in enumerate(dataloader):
        pred = classifier(adv_img.cuda())
        assert len(pred.shape) == 2
        assert len(adv_labels.shape) == 1
        num_succ_attacks += (pred.argmax(dim=1) == adv_labels.cuda()).count_nonzero().item()

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

        root = '/home/htc/kchitranshi/SCRATCH/CFE_datasets'
        os.makedirs(root, exist_ok=True)

        fname = f'celebA_{norm}'
        path = os.path.join(root, fname)

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            pathlib.Path(path).touch()

        # Fine tuning the model and setting the seed
        fine_tune_model = False
        set_seed(seed)

        # Defining transformations
        train_transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(224),
            T.ToTensor(),
        ])

        adv_train_transform = T.Compose([
            T.RandomResizedCrop(128),
            T.RandomHorizontalFlip(),
        ])

        val_transform = T.Compose([
            T.CenterCrop(178),
            T.Resize(224),
            T.ToTensor(),
        ])

        # Loading the datasets
        train_data = CelebA(root='../SCRATCH', split='train', transform=train_transform)
        train_test_data = CelebA(root='../SCRATCH', split='train', transform=val_transform)
        val_data = CelebA(root='../SCRATCH', split='valid', transform=val_transform)
        test_data = CelebA(root='../SCRATCH', split='test', transform=val_transform)

        # If fine tune model
        if fine_tune_model:
            train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                        batch_size=108, 
                                                        shuffle=True,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        train_test_dataloader = torch.utils.data.DataLoader(train_test_data,
                                                            batch_size=128,
                                                            shuffle=False,
                                                            num_workers=3,
                                                            pin_memory=True
                        )
        val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                        batch_size=128, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        # Loading validation and test datasets
        test_dataloader = torch.utils.data.DataLoader(test_data, 
                                                        batch_size=128, 
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
            state_dict = torch.load("../SCRATCH/CFE_models/celebA_aabdafe8-d805-4f62-9d76-5a50f95efbe3.pt", map_location='cpu')
            #035084af-b895-433b-bdf9-46cba06e8f51 a6676e20-4c61-45e6-97a0-54d9f9929c5a
        classifier = ModelWithNormalization(model,
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]
                    )
        if not fine_tune_model:
            classifier.load_state_dict(state_dict)

        # Fine Tuning the classifier
        classifier = self.setup(classifier)
        loss = fine_tune(classifier, train_dataloader, save_model=True) if fine_tune_model else None
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
        print(f'Accuracy of trained model on clean test data: {test_acc * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of trained model on clean val data: {val_acc * 100:.2f}%')
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
                    device= 'cuda:0',
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
        adv_attack_data = CelebA(root='../SCRATCH', split='train', transform=val_transform)
        adv_attack_loader = torch.utils.data.DataLoader(adv_attack_data, 
                                                        batch_size=256, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                            )

        # Conducting the adversarial attacks/CFEs
        print("Conducting Adversarial Attacks/CFEs.")
        avg_L2_norms = []
        for _, (data, _) in tqdm(enumerate(adv_attack_loader)):

            # Generating target labels            
            labels = generate_adv_labels(data.shape[0], "cuda:0")

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
                           lam_steps=2,
                           L0=1e-5 # changed from 1e-4 to 1e-3
                )
                adv_data = cfe_atk.get_CFs_natural_binary(data.cuda(), labels.unsqueeze(1).cuda())

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
        adv_loss = fine_tune(adv_classifier, adv_dataloader, train=True, save_model=False)
        freeze(adv_classifier)
        adv_classifier.eval()

        # Checking its accuracy on validation/test sets
        adv_acc_for_natural_train_test = test(adv_classifier, train_test_dataloader)
        adv_acc_for_natural_test = test(adv_classifier, test_dataloader)
        adv_acc_for_natural_val = test(adv_classifier, val_dataloader)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean train data: {adv_acc_for_natural_train_test * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean test data: {adv_acc_for_natural_test * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean val data: {adv_acc_for_natural_val * 100:.2f}%')
        print('-'*60)

        print('='*60)
        print('='*60)

        save_data = {
            'classifier': classifier,
            'adv_data': adv_dataset['imgs'],
            'adv_labels': adv_dataset['labels'],
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
