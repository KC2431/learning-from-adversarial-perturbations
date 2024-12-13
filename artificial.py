import argparse
import os
import pathlib
from typing import Any, Dict, Literal

import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from utils.attacks import PGDL0, PGDL2, PGDLinf
from utils.classifiers import OneHiddenNet
from utils.utils import all_elements_in_targets, freeze, set_seed
from utils.cfe import APG0_CFE
from utils.l1_mad import L1_MAD

def calc_loss(outs: Tensor, labels: Tensor) -> Tensor:
    assert outs.shape == labels.shape
    assert len(outs.shape) == len(labels.shape) == 1
    return (- outs * labels).exp()
    

class BinaryPGDL0(PGDL0):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDL2(PGDL2):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDLinf(PGDLinf):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)


def generate_data(
    mode: Literal['uniform', 'gauss'], 
    n: int, 
    dim: int, 
    device: torch.device,
) -> Tensor:
    if mode == 'uniform':
        return 2 * torch.rand(n, dim, device=device) - 1
    elif mode == 'gauss':
        return torch.normal(0, 1, (n, dim), device=device)
    else:
        raise ValueError(mode)
    

def generate_label(n: int, device: torch.device) -> Tensor:
    return 2 * torch.randint(0, 2, (n,), device=device) - 1
    

def train(classifier: OneHiddenNet, data: Tensor, labels: Tensor) -> float: # type: ignore
    assert all_elements_in_targets(labels, [-1, 1])

    optim = SGD(classifier.parameters(), 0.01, 0.9)
    scheduler = ReduceLROnPlateau(optim)

    epochs = 100000
    for epoch in tqdm(range(epochs), mininterval=120):
        outs = classifier(data)
        losses = calc_loss(outs, labels)
        loss = losses.mean()

        optim.zero_grad(True)
        loss.backward()
        optim.step()
        scheduler.step(loss)

        if epoch == epochs - 1:
            return loss.item()
        

@torch.no_grad()
def test(classifier: OneHiddenNet, data: Tensor, labels: Tensor) -> Tensor:
    assert all_elements_in_targets(labels, [-1, 1])
    return ((classifier(data) * labels) > 0).count_nonzero() / len(labels)


def to_cpu(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = v.cpu()
        elif isinstance(v, torch.nn.Module):
            d[k] = v.cpu().state_dict()
    return d


class Main(LightningLite):
    def run(
        self, 
        in_dim: int, 
        hidden_dim: int,
        n_sample: int,
        n_noise_sample: int,
        norm: Literal['L0', 'L2', 'Linf'],
        mode: Literal['uniform', 'gauss'],
        perturbation_constraint: float,
        seed: int,
    ) -> None:
        
        root = '/home/htc/kchitranshi/SCRATCH/artificial'
        os.makedirs(root, exist_ok=True)

        fname = f'{in_dim}_{hidden_dim}_{n_sample}_{n_noise_sample}_{norm}_{mode}_{perturbation_constraint}_{seed}'
        path = os.path.join(root, fname)

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            pathlib.Path(path).touch()

        set_seed(seed)

        classifier = self.setup(OneHiddenNet(in_dim, hidden_dim))
        data = generate_data(mode, n_sample, in_dim, self.device)
        labels = generate_label(n_sample, self.device)
        
        loss = train(classifier, data, labels)
        freeze(classifier)
        classifier.eval()
        acc = test(classifier, data, labels)
        noise_data = generate_data(mode, n_noise_sample, in_dim, self.device)
        target_labels = generate_label(n_noise_sample, self.device)
        noise_classifier = self.setup(OneHiddenNet(in_dim, hidden_dim))

        noise_loss = train(noise_classifier, noise_data, target_labels)
        freeze(noise_classifier)
        noise_classifier.eval()
        noise_acc = test(noise_classifier, noise_data, target_labels)
        noise_acc_for_natural = test(noise_classifier, data, labels)
        
        if mode == 'uniform':
            data_range = (-1, 1)
        elif mode == 'gauss':
            data_range = (-float('inf'), float('inf'))
        else:
            raise ValueError(mode)

        if norm == 'L0':
            steps = int(in_dim * perturbation_constraint)
            #atk = BinaryPGDL0(classifier, steps, data_range=data_range)
            """
            print("Using APG0_CFE")
            atk = APG0_CFE(model=classifier.to(noise_data.device), 
                                mins=torch.tensor(10), # Just some random value 
                                maxs=torch.tensor(10), # Just some random value
                                numclasses=2, 
                                range_min=None, 
                                range_max=None,
                                beta=25,
                                L0=1e-2,
                                lam0=1.0,
                                c=0.0,
                                prox='zero',
                                linesearch=False, 
                                iters=steps,
                                scale_model=False,
                                verbose=False,
                                lam_steps=10,
            )
            """
            atk = L1_MAD(
                model=classifier,
                max_image_range = 1.0,
                min_image_range = 0.0, 
                optimizer = torch.optim.Adam, 
                iters=steps, 
                lamb=1e-2,
                mode=mode,
                device= 'cuda:0',
            )
        elif norm == 'L2':
            atk = BinaryPGDL2(classifier, 100, perturbation_constraint, data_range)
        elif norm == 'Linf':
            atk = BinaryPGDLinf(classifier, 100, perturbation_constraint, data_range)
        else:
            raise ValueError(norm)

        if not isinstance(atk,L1_MAD):
            adv_data = atk(noise_data, target_labels)
        else:
            """
            maxs = torch.tensor(data_range[1]).repeat(noise_data.view(noise_data.shape[0], -1).size(-1))[None,...]
            mins = torch.tensor(data_range[0]).repeat(noise_data.view(noise_data.shape[0], -1).size(-1))[None,...]
            
            atk.maxs = maxs.to(noise_data.device)
            atk.mins = mins.to(noise_data.device)

            atk.range_min = atk.mins.clone().view(atk.mins.size(0), -1).to(noise_data.device)
            atk.range_max = atk.maxs.clone().view(atk.maxs.size(0), -1).to(noise_data.device)
            adv_data = atk.get_CFs(noise_data, target_labels.unsqueeze(1), mode=mode)
            """
            atk.min_image_range = data_range[0]
            atk.max_image_range = data_range[1]
            adv_data = atk.get_perturbations(noise_data, target_labels.unsqueeze(1))

        adv_data = self.to_device(adv_data)

        adv_classifier = self.setup(OneHiddenNet(in_dim, hidden_dim))

        adv_loss = train(adv_classifier, adv_data, target_labels)
        freeze(adv_classifier)
        adv_classifier.eval()
        adv_acc = test(adv_classifier, adv_data, target_labels)
        adv_acc_for_natural = test(adv_classifier, data, labels)

        save_data = {
            'classifier': classifier,
            'data': data,
            'labels': labels,
            'loss': loss,
            'acc': acc,
            'noise_data': noise_data,
            'target_labels': target_labels,
            'noise_classifier': noise_classifier,
            'noise_loss': noise_loss,
            'noise_acc': noise_acc,
            'noise_acc_for_natural': noise_acc_for_natural,
            'adv_data': adv_data,
            'adv_classifier': adv_classifier,
            'adv_loss': adv_loss,
            'adv_acc': adv_acc,
            'adv_acc_for_natural': adv_acc_for_natural,
        }
        to_cpu(save_data)
        torch.save(save_data, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dim', type=int)
    parser.add_argument('hidden_dim', type=int)
    parser.add_argument('n_sample', type=int)
    parser.add_argument('n_noise_sample', type=int)
    parser.add_argument('norm', choices=('L0', 'L2', 'Linf'))
    parser.add_argument('mode', choices=('uniform', 'gauss'))
    parser.add_argument('perturbation_constraint', type=float)
    parser.add_argument('seed', type=int)
    parser.add_argument('devices', nargs='+', type=int)
    args = parser.parse_args()

    lite_kwargs = {
        'accelerator': 'gpu',
        'strategy': 'ddp_find_unused_parameters_false',
        'devices': args.devices,
        'precision': 32,
    }
    
    Main(**lite_kwargs).run(
        args.in_dim, 
        args.hidden_dim,
        args.n_sample,
        args.n_noise_sample,
        args.norm,
        args.mode,
        args.perturbation_constraint,
        args.seed,
    )
