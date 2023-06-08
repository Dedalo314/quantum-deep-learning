import logging

from torch import Tensor, nn, no_grad, topk, multinomial, cat, max, inference_mode, argmax, optim
import torchmetrics
import lightning.pytorch as pl
from einops import rearrange
from tqdm.notebook import tqdm

from qdl.utils.import_class import import_class

logger = logging.getLogger(__name__)

class LightningClassifier(pl.LightningModule):
    r"""
    Classifier abstraction for different models

    The model passed to the init is used in the forward.
    """
    def __init__(self, cfg) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
        self._conf = cfg
        self.learning_rate = cfg.optim.lr
        self.model_class = import_class(cfg.model_class)
        self.generator = self.model_class(self._conf)
        self.train_accuracy = torchmetrics.Accuracy(
            task=cfg.metrics.task,
            num_classes=cfg.metrics.num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task=cfg.metrics.task,
            num_classes=cfg.metrics.num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task=cfg.metrics.task,
            num_classes=cfg.metrics.num_classes
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.generator(x)

    def configure_optimizers(self):
        optimizer = self.generator.configure_optimizers(
            self._conf.optim.weight_decay,
            self.learning_rate,
            (self._conf.optim.beta1, self._conf.optim.beta2)
        )
        if self._conf.optim.use_cosine_annealing:
            stepping_batches = self.trainer.estimated_stepping_batches
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        return self._shared_eval(train_batch, batch_idx, "train")

    def validation_step(self, val_batch, batch_idx):
        self._shared_eval(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        self._shared_eval(test_batch, batch_idx, "test")

    def on_after_backward(self):
        if self._conf.logger.log_grads:
            global_step = self.global_step
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
        else:
            pass

    def _shared_eval(self, batch, batch_idx, prefix):
        X, y = batch
        logits = self(X)
        loss = nn.functional.cross_entropy(logits, y)
        if prefix == "train":
            self.train_accuracy(logits, y)
            self.log(
                f"Accuracy/{prefix}",
                self.train_accuracy,
                on_step=True,
                on_epoch=False
            )
        elif prefix == "val":
            self.val_accuracy(logits, y)
            self.log(
                f"Accuracy/{prefix}",
                self.val_accuracy,
                on_step=False,
                on_epoch=True
            )
        elif prefix == "test":
            self.test_accuracy(logits, y)
            self.log(
                f"Accuracy/{prefix}",
                self.test_accuracy,
                on_step=True,
                on_epoch=True
            )
        else:
            raise NotImplementedError
        self.log(f"Loss/{prefix}", loss)
        return loss
