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
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np

from utils.classifiers.binary_models import BinaryTrainedResNet
from utils.datasets import BinaryDataset, SequenceDataset, SpuCoDogsDataset
from utils.attacks import PGDL0, PGDL2, PGDLinf
from utils.Models import VAE, train_vae_v1
from utils.utils import freeze, set_seed
from utils.gdpr_cfe import GDPR_CFE
from utils.scfe import APG0_CFE, APG0_CFE_VAE
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
    

def fine_tune(classifier, dataloader, fname, method='Orig', save_model=True, seed=10, model_save_dir='../SCRATCH/CFE_models') -> float: # type: ignore
    
    epochs = 60
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
            outs = classifier(imgs.to(classifier.device))
            losses = calc_loss(outs, labels.to(classifier.device))
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
                if method == 'Orig':
                    torch.save(classifier.state_dict(), f'{model_save_dir}/SpuCO_dogs_trained_model_seed_{seed}.pt')
                else:
                    torch.save(classifier.state_dict(), f'{model_save_dir}/{fname}.pt')
            return loss.item()


@torch.no_grad()
def test(classifier, dataloader) -> Tensor:
    num_samples = 0
    num_correct = 0
    
    for _, (imgs, labels) in enumerate(dataloader):
        num_big_dog = 0 
        num_small_dog = 0 
        assert len(labels.shape) == 1
        num_samples += len(labels)

        output = classifier(imgs.to(classifier.device))
        num_correct += (output.argmax(dim=1) == labels.to(classifier.device)).count_nonzero().item()
        num_big_dog += (output.argmax(dim=1) == labels.to(classifier.device)).logical_and(labels.to(classifier.device) == 0).count_nonzero().item()
        num_small_dog += (output.argmax(dim=1) == labels.to(classifier.device)).logical_and(labels.to(classifier.device) == 1).count_nonzero().item()
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

def get_percentage_pert_data(pert_data, percentage=20):
    total_samples = int(len(pert_data) * percentage / 100)
    idxs_array = [i for i in range(len(pert_data))]
    random_samples_idxs = np.random.choice(idxs_array, size=total_samples, replace = False)

    pert_data_subset = torch.utils.data.Subset(pert_data, random_samples_idxs)
    return pert_data_subset

class Main(LightningLite):
    def run(self,
        norm: Literal['GDPR_CFE', 'SCFE', 'L2', 'Linf'],
        comb_nat_pert: Literal['yes', 'no'],
        percentage: int,
        seed: int,
    ) -> None:

        print(f'UUID: {UUID}')

        data_dir = '../SCRATCH'
        model_save_dir = '../SCRATCH/CFE_models'
        root = '/home/htc/kchitranshi/SCRATCH/CFE_datasets'
        os.makedirs(root, exist_ok=True)

        comb_nat_pert = True if comb_nat_pert == 'yes' else False

        fname = f'SpuCO_dogs_{norm}_seed_{seed}_percentage_{percentage}' if comb_nat_pert else f'SpuCO_dogs_{norm}_seed_{seed}'
        path = os.path.join(root, fname)

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            pathlib.Path(path).touch()

        # Fine tuning the model and setting the seed
        fine_tune_model = False
        train_vae = True
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
        train_data = SpuCoDogsDataset(f'{data_dir}/spuco_dogs/train', transform=train_transform)
        train_test_data = SpuCoDogsDataset(f'{data_dir}/spuco_dogs/train', transform=val_transform)
        val_data = SpuCoDogsDataset(f'{data_dir}/spuco_dogs/val', transform=val_transform)
        test_data = SpuCoDogsDataset(f'{data_dir}/spuco_dogs/test', transform=val_transform)

        train_data = BinaryDataset(train_data, 'spuco_dogs')
        # If fine tune model
        if fine_tune_model:
            train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                        batch_size=108, 
                                                        shuffle=True,
                                                        num_workers=3,
                                                        pin_memory=True
                        )

        # Loading validation and test datasets
        val_data = BinaryDataset(val_data, 'spuco_dogs')
        val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                        batch_size=128, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        test_data = BinaryDataset(test_data, 'spuco_dogs')
        test_dataloader = torch.utils.data.DataLoader(test_data, 
                                                        batch_size=128, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        train_test_data = BinaryDataset(train_test_data, 'spuco_dogs')
        train_test_dataloader = torch.utils.data.DataLoader(train_test_data,
                                                            batch_size=128,
                                                            shuffle=False,
                                                            num_workers=3,
                                                            pin_memory=True
                        )
        # Loading the model
        model = BinaryTrainedResNet('resnet50', num_classes=2)
        if fine_tune_model:
            model.train()
        else:
            print(f"Loading Pre-trained Model.")
            state_dict = torch.load(f"{model_save_dir}/SpuCO_dogs_trained_model_seed_{seed}.pt", map_location='cpu')
        classifier = ModelWithNormalization(model,
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]
                    )
        if not fine_tune_model:
            classifier.load_state_dict(state_dict)

        # Fine Tuning the classifier
        classifier = self.setup(classifier)
        loss = fine_tune(classifier, train_dataloader, fname=fname, method='Orig', save_model=True, model_save_dir=model_save_dir) if fine_tune_model else None
        freeze(classifier)
        classifier.eval()

        # Calculating accuracy on validation and test sets
        val_acc = test(classifier, val_dataloader)
        test_acc = test(classifier, test_dataloader)
        train_test_acc = test(classifier, train_test_dataloader)

        print('-'*60)
        print(f'Accuracy of trained model on clean train data: {train_test_acc * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of trained model on clean validation data: {val_acc * 100:.2f}%')
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
            atk = APG0_CFE_VAE

            vae_0 = VAE()
            vae_1 = VAE()

            vae_0 = self.setup(vae_0)
            vae_1 = self.setup(vae_1)
            
            if train_vae:       

                class_0_indices = [i for i in range(len(train_data)) 
                   if train_data[i][1] == 0]
                
                class_1_indices = [i for i in range(len(train_data)) 
                if train_data[i][1] == 1]

                train_data_class_0 = torch.utils.data.Subset(train_data, class_0_indices)
                train_data_class_1 = torch.utils.data.Subset(train_data, class_1_indices)

                train_dataloader_class_0 = torch.utils.data.DataLoader(train_data_class_0, 
                                                            batch_size=256, 
                                                            shuffle=True,
                                                            num_workers=3,
                                                            pin_memory=True
                            )
                    
                train_dataloader_class_1 = torch.utils.data.DataLoader(train_data_class_1, 
                                                            batch_size=256, 
                                                            shuffle=True,
                                                            num_workers=3,
                                                            pin_memory=True
                            )

                vae_0.train()
                vae_1.train()
                
                print("Training VAEs now for both classes.")

                optim_vae_0 = torch.optim.Adam(params=vae_0.parameters(),
                                              lr=1e-4,
                                              weight_decay=1e-4
                            )
                optim_vae_1 = torch.optim.Adam(params=vae_1.parameters(),
                                              lr=1e-4,
                                              weight_decay=1e-4
                            )

                vae_0_log_dict = train_vae_v1(
                    model=vae_0,
                    num_epochs=100,
                    optimizer=optim_vae_0,
                    train_loader=train_dataloader_class_0,
                    logging_interval=50,
                    device=vae_0.device,
                    reconstruction_term_weight=0.05,
                    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_vae_0, min_lr=1e-7),
                    save_model=f'{model_save_dir}/SpuCO_dogs_class_0_seed_{seed}.pt'
                )
                
                vae_1_log_dict = train_vae_v1(
                    model=vae_1,
                    num_epochs=100,
                    optimizer=optim_vae_1,
                    train_loader=train_dataloader_class_1,
                    logging_interval=50,
                    device=vae_1.device,
                    reconstruction_term_weight=0.05,
                    lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optim_vae_1, min_lr=1e-7),
                    save_model=f'{model_save_dir}/SpuCO_dogs_class_1_seed_{seed}.pt'
                )

            else:
                print(f"Loading pre-trained VAEs for both classes.")

                vae_0_state_dict = torch.load(f'{model_save_dir}/SpuCO_dogs_class_0_seed_{seed}.pt')
                vae_1_state_dict = torch.load(f'{model_save_dir}/SpuCO_dogs_class_1_seed_{seed}.pt')

                vae_0.load_state_dict(vae_0_state_dict)
                vae_1.load_state_dict(vae_1_state_dict)

            vae_0.eval()
            vae_1.eval()

            vaes = [vae_0, vae_1]

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

        adv_dataset = {'imgs': [], 'labels': []}

        # Intialising the data to be attacked
        adv_attack_data = SpuCoDogsDataset(f'{data_dir}/spuco_dogs/train', transform=val_transform)
        adv_attack_data = BinaryDataset(adv_attack_data, 'spuco_dogs')

        if comb_nat_pert:
            adv_attack_data = get_percentage_pert_data(adv_attack_data, percentage=percentage)

        adv_attack_loader = torch.utils.data.DataLoader(adv_attack_data, 
                                                        batch_size=256, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                            )

        # Conducting the adversarial attacks/CFEs
        avg_L2_norms = []
        avg_L0_norms = []
        for _, (data, _) in tqdm(enumerate(adv_attack_loader)):

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
                           lam0=5e-2,
                           vaes=vaes,
                           theta=0.1,
                           lam_steps=4,
                           L0=1e-4, # changed from 1e-3 to 1e-4
                           beta=2e-7, # Less beta for less sparsity
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
            avg_L0_norms.append(torch.logical_not(torch.isclose(adv_data.cpu(), data)).float().sum(dim=(1,2,3)).mean().item())
            adv_dataset['imgs'].append(adv_data.cpu())
            adv_dataset['labels'].append(labels.cpu())
        
        # Concating the adversarial/CFE data
        adv_dataset['imgs'] = torch.cat(adv_dataset['imgs'])
        adv_dataset['labels'] = torch.cat(adv_dataset['labels'])

        # Defining the dataloaders for the adversarial/CFE data
        adv_data = SequenceDataset(adv_dataset['imgs'], adv_dataset['labels'],x_transform=None)
        
        if comb_nat_pert:
            adv_data = torch.utils.data.ConcatDataset([train_data, adv_data])

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
        print(f'The attack success rate is {attack_succ_rate * 100:.2f} with an avg L2 norm of {np.mean(avg_L2_norms).item():.2f} and avg L0 norm of {np.mean(avg_L0_norms).item():.2f}')
        print('-'*60)

        # Intialising the Adversarial model
        adv_model = BinaryTrainedResNet(resnet_type='resnet50', num_classes=2)
        adv_model.train()
        adv_classifier = ModelWithNormalization(adv_model,
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]
                    )
        adv_classifier = self.setup(adv_classifier)
        adv_loss = fine_tune(adv_classifier, adv_dataloader, fname=fname, method=norm, save_model=True, model_save_dir=model_save_dir)
        freeze(adv_classifier)
        adv_classifier.eval()

        # Checking its accuracy on validation/test sets
        adv_acc_for_natural_val = test(adv_classifier, val_dataloader)
        adv_acc_for_natural_test = test(adv_classifier, test_dataloader)
        adv_acc_for_natural_train_test = test(adv_classifier, train_test_dataloader)

        print('-'*60)       
        print(f'Accuracy of Adversarially trained model on clean train data: {adv_acc_for_natural_train_test * 100:.2f}%')
        print('-'*60)

        print('-'*60)       
        print(f'Accuracy of Adversarially trained model on clean validation data: {adv_acc_for_natural_val * 100:.2f}%')
        print('-'*60)

        print('-'*60)
        print(f'Accuracy of Adversarially trained model on clean test data: {adv_acc_for_natural_test * 100:.2f}%')
        print('-'*60)

        print('='*60)
        print('='*60)

        save_data = {
            'classifier': classifier,
            'adv_data': adv_dataset['imgs'],
            'adv_labels': adv_dataset['labels'],
            'clean_fine_tune_loss': loss if fine_tune_model else None,
            'clean_val_acc': val_acc,
            'adv_classifier': adv_classifier,
            'adv_train_loss': adv_loss,
            'adv_acc_for_natural_val': adv_acc_for_natural_val,
            'adv_acc_for_natural_test': adv_acc_for_natural_test,
            'avg_L2_norms': avg_L2_norms
        }
        to_cpu(save_data)
        torch.save(save_data, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('norm', choices=('GDPR_CFE', 'SCFE', 'L2', 'Linf'))
    parser.add_argument('comb_nat_pert', choices=['yes', 'no'])
    parser.add_argument('percentage', type=int)
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
        args.comb_nat_pert,
        args.percentage,
        args.seed,
    )
