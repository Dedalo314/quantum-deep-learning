import logging

from torch import Tensor, nn, no_grad, topk, multinomial, cat, max, inference_mode, argmax
import lightning.pytorch as pl
from einops import rearrange
from tqdm.notebook import tqdm

from qdl.utils.import_class import import_class

logger = logging.getLogger(__name__)

class LightningGenerator(pl.LightningModule):
    r"""
    Generator abstraction for different models

    The model passed to the init is used in the forward.
    """
    def __init__(self, cfg) -> None:
        super(LightningGenerator, self).__init__()
        self.save_hyperparameters(cfg)
        self._conf = cfg
        self.learning_rate = cfg.lr
        self.model_class = import_class(cfg.model_class)

    def forward(self, x: Tensor) -> Tensor:
        return self.generator(x)

    @inference_mode()
    def generate_max_prob(self, idx, max_new_tokens):
        """
        Based on NanoGPT

        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        The token with highest probability is always selected.
        """
        for _ in tqdm(range(max_new_tokens)):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self._conf.block_size else idx[:, -self._conf.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step
            logits = logits[:, -1, :]
            # take token with highest probability (highest logit)
            idx_next = argmax(logits, dim=-1, keepdim=True)
            # append sampled index to the running sequence and continue
            idx = cat((idx, idx_next + 10), dim=1)
        return idx

    @inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Based on NanoGPT

        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in tqdm(range(max_new_tokens)):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self._conf.block_size else cat(
                (rearrange(idx[:, 0], "b -> b 1"), idx[:, -self._conf.block_size:]), dim=1
            )
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = cat((idx, idx_next + 10), dim=1)

        return idx

    def configure_sharded_model(self) -> None:
        self.generator = self.model_class(self._conf)

    def configure_optimizers(self):
        optimizer = self.generator.configure_optimizers(
            self._conf.weight_decay,
            self.learning_rate,
            (self._conf.beta1, self._conf.beta2)
        )
        return optimizer

    def training_step(self, train_batch, batch_idx):
        return self._shared_eval(train_batch, batch_idx, "train")

    def validation_step(self, val_batch, batch_idx):
        self._shared_eval(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        self._shared_eval(test_batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        X, y = batch
        logits = self(X)
        logger.debug(f"{logits.shape=}")
        logger.debug(f"{y.shape=}")
        logger.debug(f"{y=}")
        loss = nn.functional.cross_entropy(rearrange(logits, "b n h -> (b n) h"), rearrange(y, "b n -> (b n)"))
        self.log(f"Loss/{prefix}", loss)
        return loss
