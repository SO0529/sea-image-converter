import random

import pytorch_lightning as pl
import torch
from utils import utils


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def set_hparams(self):
        # for LightningDataModule
        self.hparams["batch_size"] = self.cfg.batch_size
        self.hparams["num_workers"] = self.cfg.num_workers
        self.hparams["prefetch_factor"] = self.cfg.prefetch_factor

    # https://github.com/PyTorchLightning/pytorch-lightning/issues/2484
    @property
    def batch_size(self):
        return self.hparams.batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.hparams.batch_size = batch_size

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            DatasetPair(self.cfg, phase="train"),
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            DatasetPair(self.cfg, phase="val"),
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
            prefetch_factor=self.hparams.prefetch_factor,
        )
        return val_loader


class DatasetPair(object):
    def __init__(self, cfg, phase):
        super(DatasetPair, self).__init__()
        self.cfg = cfg
        self.phase = phase
        self.n_channels = cfg.input_shape[0]
        self.patch_height = cfg.input_shape[1]
        self.patch_width = cfg.input_shape[2]

        # get paths
        if phase == "train":
            self.input_paths = utils.get_image_paths(cfg.train_data.dataroot_IN)
            self.gt_paths = utils.get_image_paths(cfg.train_data.dataroot_GT)
        elif phase == "val":
            self.input_paths = utils.get_image_paths(cfg.val_data.dataroot_IN)
            self.gt_paths = utils.get_image_paths(cfg.val_data.dataroot_GT)

    def __getitem__(self, index):
        # get gt and input image
        gt_path = self.gt_paths[index]
        gt_img = utils.imread_uint(gt_path, self.n_channels)
        gt_img = utils.uint2single(gt_img)

        input_path = self.input_paths[index]
        input_img = utils.imread_uint(input_path, self.n_channels)
        input_img = utils.uint2single(input_img)

        assert (
            gt_img.shape[0] == input_img.shape[0] and gt_img.shape[1] == input_img.shape[1]
        ), "Error: Unmatch image shape."

        # crop following input_shape
        h, w, _ = input_img.shape
        rnd_h = random.randint(0, max(0, h - self.patch_height))
        rnd_w = random.randint(0, max(0, w - self.patch_width))
        input_img = input_img[rnd_h: rnd_h + self.patch_height, rnd_w: rnd_w + self.patch_width, :]
        gt_img = gt_img[rnd_h: rnd_h + self.patch_height, rnd_w: rnd_w + self.patch_width, :]

        # augmentation - flip and/or rotate
        if self.phase == "train":
            mode = random.randint(0, 3)
            input_img, gt_img = utils.augment_img(input_img, mode=mode), utils.augment_img(gt_img, mode=mode)

        # HWC to CHW, numpy to tensor
        input_img, gt_img = utils.single2tensor3(input_img), utils.single2tensor3(gt_img)

        return {"input": input_img, "gt": gt_img, "input_path": input_path, "gt_path": gt_path}

    def __len__(self):
        return len(self.gt_paths)
