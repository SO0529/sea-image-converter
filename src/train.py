import os
import warnings
import datetime
from omegaconf import OmegaConf

import pytorch_lightning as pl
import torch
from data.dataset import DataModule
from pytorch_lightning import seed_everything

from nets.sea_image_conveter import SeaImageConverter

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


if __name__ == "__main__":
    train()
