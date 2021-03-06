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
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor

from nets.sea_image_conveter import SeaImageConverter
from utils.logger import ValImageLogger
from utils import utils

warnings.filterwarnings("ignore")
CONFIG_FILE = "./config/train.yaml"
DEBUG = False
logger_type = "wandb"


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
        cfg.trainer.args.max_epochs = 5
        cfg.trainer.args.check_val_every_n_epoch = 5
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
    model.train_setup()

    # resume
    if cfg.resume is not None:
        resume_from_checkpoint = cfg.resume
    else:
        resume_from_checkpoint = None

    # wandb logger
    if logger_type == "wandb":
        logger = WandbLogger(
            project=cfg.project,
            name=cfg.name,
            log_model=True
            )
        log_params = utils.get_log_param(cfg)
        logger.log_hyperparams(params=log_params)

        wandb_image_logger = ValImageLogger(
            val_dataloader=dm.val_dataloader()
            )
    else:
        logger = pl_loggers.TensorBoardLogger(cfg.save_dir)

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

    # trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_checkpoint, early_stopping, wandb_image_logger, lr_monitor],
        # callbacks=[model_checkpoint, early_stopping, lr_monitor],
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

    # test using best model
    test_dataloader = dm.test_dataloader()
    trainer.test(dataloaders=test_dataloader, ckpt_path=model_checkpoint.best_model_path)


if __name__ == "__main__":
    train()
