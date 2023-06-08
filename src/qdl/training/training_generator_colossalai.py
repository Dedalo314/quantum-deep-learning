"""
Training script using hydra.cc and PL.
"""
import logging

import hydra
import lightning.pytorch as pl
from lightning_colossalai import ColossalAIStrategy
import torch

from qdl.utils.import_class import import_class
from qdl.models.LightningGenerator import LightningGenerator
from qdl.data.LightningDataModule import LightningDataModule

# Default hydra logger
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    data_module = LightningDataModule(cfg=cfg.data)

    logger.info(f"Batch size: {cfg.data.train.batch_size}")

    generator = LightningGenerator(cfg=cfg.model)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.epochs,
        default_root_dir=cfg.trainer.default_root_dir,
        strategy=ColossalAIStrategy(
            placement_policy="auto"
        ),
        num_sanity_val_steps=0,
        limit_val_batches=10,
        precision="16-mixed",
        log_every_n_steps=cfg.trainer.log_every_n_steps
    )
    trainer.fit(generator, data_module)


try:
    main()
except Exception as ex:
    logger.exception(ex)
    raise
