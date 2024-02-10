from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import cross_entropy, normalize

from .utils import freeze


class PGD(ABC):
    def __init__(self, classifier: Module, steps: int) -> None:
        self.classifier = classifier
        self.steps = steps


    @staticmethod
    @torch.no_grad()
    def _replace_best(
        advs: Tensor, 
        losses: Tensor, 
        best_advs: Optional[Tensor], 
        best_losses: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:

        if best_advs is None and best_losses is None:
            best_advs = advs.detach().cpu().clone()
            best_losses = losses.detach().cpu().clone()

        elif best_advs is not None and best_losses is not None:
            replace = best_losses > losses.detach().cpu().clone()
            best_advs[replace] = advs[replace].detach().cpu().clone()
            best_losses[replace] = losses[replace].detach().cpu().clone()

        else:
            raise ValueError(best_advs, best_losses)
            
        return best_advs, best_losses
    

    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return cross_entropy(outs, targets, reduction='none')


    @abstractmethod
    @torch.no_grad()
    def _project(self, advs: Tensor, grads: Tensor, orig_imgs: Tensor) -> Tensor:
        raise NotImplementedError


    def __call__(self, imgs: Tensor, targets: Tensor) -> Tensor: # type: ignore
        freeze(self.classifier)
        self.classifier.eval()

        orig_imgs = imgs.detach().clone()
        advs = orig_imgs.clone()

        best_advs, best_losses = None, None
        for i in range(self.steps+1):
            advs = advs.detach().requires_grad_(True)

            outs = self.classifier(advs)
            losses = self._calc_loss(outs, targets)

            best_advs, best_losses = self._replace_best(advs, losses, best_advs, best_losses)

            if i == self.steps:
                return best_advs

            loss = losses.sum()
            grads, = torch.autograd.grad(loss, advs)

            advs = self._project(advs, grads, orig_imgs)


class PGDL0(PGD):
    def __init__(
        self, 
        classifier: Module, 
        steps: int, 
        step_size: float = 0.3,
        data_range: Tuple[float, float] = (0, 1),
    ) -> None:
        self.classifier = classifier
        self.steps = steps
        self.step_size = step_size
        self.data_range = data_range


    @torch.no_grad()
    def _project(self, advs: Tensor, grads: Tensor, orig_imgs: Tensor) -> Tensor:
        s = advs.shape
        
        flatten_diff = (advs - orig_imgs).flatten(1)
        changed_pixels = flatten_diff != 0
        not_changed_pixels = flatten_diff == 0

        flatten_grads = grads.flatten(1)
        max_grad_indices = (flatten_grads * not_changed_pixels).abs().argmax(1)
        changed_pixels[range(s[0]), max_grad_indices] = True
        masked_flatten_grads = flatten_grads * changed_pixels

        normalized_masked_flatten_grads = normalize(masked_flatten_grads, p=float('inf'))

        normalized_masked_grads = normalized_masked_flatten_grads.view(s)
        advs = advs - normalized_masked_grads * self.step_size

        return torch.clamp(advs, self.data_range[0], self.data_range[1])
    

class PGDL2(PGD):
    p = 2

    def __init__(
        self, 
        classifier: Module, 
        steps: int, 
        eps: float, 
        data_range: Tuple[float, float] = (0, 1),
    ) -> None:
        self.classifier = classifier
        self.steps = steps
        self.eps = eps
        self.step_size = eps / 5
        self.data_range = data_range


    @torch.no_grad()
    def _project(self, advs: Tensor, grads: Tensor, orig_imgs: Tensor) -> Tensor:
        assert len(grads.shape) > 1
        
        flatten_grads = grads.flatten(1)
        normalized_flatten_grads = normalize(flatten_grads, p=self.p)
        normalized_grads = normalized_flatten_grads.view(grads.shape)
        advs = advs - self.step_size * normalized_grads

        diff = advs - orig_imgs
        diff = diff.renorm(p=self.p, dim=0, maxnorm=self.eps)

        return torch.clamp(orig_imgs + diff, self.data_range[0], self.data_range[1])


class PGDLinf(PGDL2):
    p = float('inf')