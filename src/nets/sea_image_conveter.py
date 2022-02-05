import numpy as np
from collections import OrderedDict
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.rrdb_model import RRDBModel
from nets.commons import Weights_Normal, VGG19_PercepLoss


class SeaImageConverter(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # general
        self.input_shape = cfg.input_shape  # [3, 256, 144]

        # models
        self.generator = GeneratorFunieGAN(
            in_nc=cfg.input_shape[0],
            out_nc=cfg.input_shape[0],
            nf=cfg.model.rrdb.nf,
            nb=cfg.model.rrdb.nb,
            gc=cfg.model.rrdb.gc
        )
        self.discriminator = DiscriminatorFunieGAN()

        # losses
        self.adversarial_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()       # similarity loss (l1)
        self.vgg_loss = VGG19_PercepLoss()     # content loss (vgg)
        self.l1_alpha, self.vgg_alpha = 7, 3     # 7:3 (as in paper)
        self.patch = (1, self.input_shape[1]//16, self.input_shape[2]//16)  # 16x9 for 256x144

        # optimizer
        self.lr = cfg.optimizer.lr
        self.b1 = cfg.optimizer.b1
        self.b2 = cfg.optimizer.b2

        # schesuler
        _milestones = list(map(lambda x: x * self.cfg.trainer.args.max_epochs, self.cfg.scheduler.milestones))
        self.milestones = list(set(list(map(lambda x: np.round(x), _milestones))))
        self.gamma = self.cfg.trainer.scheduler.gamma

    def forward(self, x):
        """
        Return: fake image
        """
        out = self.generator(x)
        return out

    def training_step(self, batch, batch_idx, optimizer_idx):
        # get input and gt batch
        input_img = batch["input"]
        gt_img = batch["gt"]

        # train generator
        if optimizer_idx == 0:
            # generate image
            fake_img = self(input_img)

            # predict by discriminator
            fake_pred = self.discriminator(fake_img, input_img)

            # Adversarial ground truths
            valid = torch.ones(input_img.size(0), *self.patch)
            valid = valid.type_as(input_img)

            # loss
            g_loss = self.adversarial_loss(fake_pred, valid)
            l1_loss = self.l1_loss(fake_img, gt_img)
            vgg_loss = self.vgg_loss(fake_img, gt_img)
            total_g_loss = g_loss + self.l1_alpha * l1_loss + self.vgg_alpha * vgg_loss
            tqdm_dict = {"g_loss": total_g_loss}
            output = OrderedDict({"loss": total_g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(imgs.size(0), 1)
            valid = valid.type_as(imgs)

            real_loss = self.adversarial_loss(self.discriminator(imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(imgs.size(0), 1)
            fake = fake.type_as(imgs)

            fake_loss = self.adversarial_loss(self.discriminator(self(z).detach()), fake)

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        # oprimizer
        lr = self.lr
        b1 = self.b1
        b2 = self.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))

        # schesuler
        scheduler = MultiStepLR(opt_g, milestones=self.milestones, gamma=self.gamma)

        return [opt_g, opt_d], [scheduler, scheduler]
