"""
Training script using hydra.cc and PL.
"""
import logging
import random

import hydra
import lightning.pytorch as pl
import torch
import numpy as np

from qdl.utils.import_class import import_class
from qdl.models.LightningClassifier import LightningClassifier
from qdl.data.LightningDataModule import LightningDataModule

# Default hydra logger
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    logging.basicConfig(level=logging.INFO)
    data_module = LightningDataModule(cfg=cfg.data)

    logger.info(f"Batch size: {cfg.data.train.batch_size}")

    classifier = LightningClassifier(cfg=cfg.model)
    classifier.configure_sharded_model()

    lr_monitor = pl.callbacks.LearningRateMonitor()

    trainer = pl.Trainer(
        callbacks=[lr_monitor],
        **cfg.trainer
    )
    if "ckpt_path" in cfg:
        trainer.fit(classifier, data_module, ckpt_path=cfg.ckpt_path)
    else:
        trainer.fit(classifier, data_module)
    trainer.test(classifier, data_module)

try:
    main()
except Exception as ex:
    logger.exception(ex)
    raise
