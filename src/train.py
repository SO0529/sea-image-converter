import os
from glob import glob
import warnings
import datetime
from omegaconf import OmegaConf

import pytorch_lightning as pl
import torch
from data.dataset import DataModule
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from nets.sea_image_conveter import SeaImageConverter
from utils.logger import ValImageLogger
from utils import utils

warnings.filterwarnings("ignore")
CONFIG_FILE = "./config/train.yaml"
DEBUG = True


def train() -> None:
    # laod config
    cfg = OmegaConf.load(CONFIG_FILE)

    # save config
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M")
    cfg.save_dir = cfg.save_dir + "_" + current_time
    os.makedirs(cfg.save_dir, exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.save_dir}/config.yaml")

    if DEBUG:
        cfg.trainer.args.max_epochs = 3
        cfg.trainer.args.check_val_every_n_epoch = 1
        cfg.trainer.args.limit_train_batches = 2
        cfg.trainer.args.limit_val_batches = 1
        cfg.project = "debug"

    # set seed
    seed_everything(cfg.trainer.manual_seed, workers=True)

    # data module
    dm = DataModule(cfg)
    dm.set_hparams()

    # cread model
    model = SeaImageConverter(cfg)

    # resume
    if cfg.resume is not None:
        resume_from_checkpoint = cfg.resume
    else:
        resume_from_checkpoint = None

    # wandb logger
    wandb_logger = WandbLogger(
        project=cfg.project,
        name=cfg.name,
        log_model=True
        )
    wandb_logger.watch(model, log="gradients", log_freq=100)
    log_params = utils.get_log_param(cfg)
    wandb_logger.log_hyperparams(params=log_params)

    # callbacks
    early_stopping = EarlyStopping(
        monitor=cfg.model.metric,
        patience=100,
        mode=cfg.model.mode,
        verbose=True
    )
    model_checkpoint = ModelCheckpoint(
        monitor=cfg.model.metric,
        save_top_k=1,
        dirpath=f"{cfg.save_dir}/saved_model",
        filename="best_model-{epoch}-{val_loss:.4f}",
        mode=cfg.model.mode,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(
        logging_interval="step"
    )
    wandb_image_logger = ValImageLogger(
        val_dataloader=dm.val_dataloader()
        )

    # trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[model_checkpoint, early_stopping, wandb_image_logger, lr_monitor],
        resume_from_checkpoint=resume_from_checkpoint,
        **cfg.trainer.args,
    )

    # tune parameters automatically
    trainer.tune(model, dm)

    # train
    trainer.fit(model, dm)

    # save best model as .pth
    model = SeaImageConverter(cfg)
    checkpoint = [s for s in glob(f"{model_checkpoint.dirpath}/*") if "best" in s][0]
    model = model.load_from_checkpoint(checkpoint_path=checkpoint, cfg=cfg)
    torch.save(model.state_dict(), f"{model_checkpoint.dirpath}/best_model.pth")


if __name__ == "__main__":
    train()
