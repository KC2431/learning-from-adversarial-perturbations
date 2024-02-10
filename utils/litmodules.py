from typing import Any, Dict, List, Optional, Tuple, Type

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanMetric, SumMetric
from torchmetrics.classification.accuracy import MulticlassAccuracy

from .callbacks import LearningRateLogger, SaveSettings
from .utils import ModelWithNormalization


class Classification(LightningModule):
    metric_names: Tuple[str, ...] = ('count', 'loss', 'acc')

    def __init__(
        self, 
        classifier: ModelWithNormalization, 
        n_class: int,
        optim: Type[Optimizer], 
        optim_kwargs: Dict[str, Any],
        scheduler: Any = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.classifier = classifier
        self.n_class = n_class

        self.optim = optim
        self.optim_kwargs = optim_kwargs

        self.scheduler = scheduler
        self.scheduler_kwargs = scheduler_kwargs

        for prefix in ('training', 'val', 'test'):
            setattr(self, f'{prefix}_count_metric', SumMetric(nan_strategy='error'))
            setattr(self, f'{prefix}_loss_metric', MeanMetric(nan_strategy='error'))
            setattr(self, f'{prefix}_acc_metric', MulticlassAccuracy(n_class, average='micro'))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._shared_step(batch, batch_idx, 'training')

    def training_step_end(self, step_output: Dict[str, Tensor]) -> None:
        return self._shared_step_end(step_output, 'training')

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._shared_step(batch, batch_idx, 'val')

    def validation_step_end(self, step_output: Dict[str, Tensor]) -> None:
        return self._shared_step_end(step_output, 'val')

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self._shared_step(batch, batch_idx, 'test')

    def test_step_end(self, step_output: Dict[str, Tensor]) -> None:
        return self._shared_step_end(step_output, 'test')

    def _shared_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, prefix: str) -> Dict[str, Tensor]:
        imgs, labels = batch
        logits = self(imgs)
        losses = cross_entropy(logits, labels, reduction='none')
        loss = losses.mean()
        return {
            'labels': labels,
            'logits': logits,
            'losses': losses,
            'loss': loss,
        }

    def _shared_step_end(self, step_output: Dict[str, Tensor], prefix: str) -> None:
        labels = step_output['labels']
        logits = step_output['logits']
        losses = step_output['losses']

        count_metric = getattr(self, f'{prefix}_count_metric')
        loss_metric = getattr(self, f'{prefix}_loss_metric')
        acc_metric = getattr(self, f'{prefix}_acc_metric')

        count_metric(len(losses))
        loss_metric(losses)
        acc_metric(logits, labels)

    def training_epoch_end(self, outputs: Any) -> None:
        self._shared_epoch_end(outputs, 'training')
        self._shared_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs: Any) -> None:
        self._shared_epoch_end(outputs, 'test')

    def _shared_epoch_end(self, outputs: Any, prefix: str) -> None:
        d = {}
        for name in self.metric_names:
            d[f'{prefix}_{name}'] = getattr(self, f'{prefix}_{name}_metric')
        self.log_dict(d, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> Any:
        optim = self.optim(self.parameters(), **self.optim_kwargs)
        if self.scheduler is None and self.scheduler_kwargs is None:
            return optim
        elif self.scheduler is not None and self.scheduler_kwargs is not None:
            scheduler = self.scheduler(optim, **self.scheduler_kwargs)
            if isinstance(scheduler, ReduceLROnPlateau):
                return {'optimizer': optim, 'lr_scheduler': scheduler, 'monitor': 'training_loss'}
            else:
                return [optim], [scheduler]
        else:
            raise ValueError(self.scheduler, self.scheduler_kwargs)

    def configure_callbacks(self) -> List[Callback]:
        return [
            SaveSettings(), 
            ModelCheckpoint(monitor='val_acc', mode='max'), 
            LearningRateLogger(),
        ]