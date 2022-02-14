import os
import numpy as np
from collections import OrderedDict
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import MultiStepLR

from nets.funiegan import GeneratorFunieGAN, DiscriminatorFunieGAN
from nets.commons import VGG19_PercepLoss
from utils import utils
from utils.measure import Measure


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

    def train_setup(self):
        # losses
        self.adversarial_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()       # similarity loss (l1)
        self.vgg_loss = VGG19_PercepLoss()     # content loss (vgg)
        self.l1_alpha, self.vgg_alpha = 7, 3     # 7:3 (as in paper)
        self.patch = (1, self.input_shape[1]//16, self.input_shape[2]//16)  # 16x9 for 256x144

        # optimizer
        self.lr = self.cfg.trainer.optimizer.lr
        self.b1 = self.cfg.trainer.optimizer.b1
        self.b2 = self.cfg.trainer.optimizer.b2

        # schesuler
        _milestones = list(map(lambda x: x * self.cfg.trainer.args.max_epochs, self.cfg.trainer.scheduler.milestones))
        self.milestones = list(set(list(map(lambda x: np.round(x), _milestones))))
        self.gamma = self.cfg.trainer.scheduler.gamma

        # val
        self.val_pre_perceptual_loss = torch.tensor(1.0)
        self.measure = Measure()

        self.save_dir = self.cfg.save_dir

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
            self.log("g_loss", total_g_loss, prog_bar=True)
            return output

        # train discriminator
        if optimizer_idx == 1:
            fake_img = self(input_img)
            real_pred = self.discriminator(gt_img, input_img)
            fake_pred = self.discriminator(fake_img, input_img)

            valid = torch.ones(input_img.size(0), *self.patch)
            valid = valid.type_as(input_img)
            real_loss = self.adversarial_loss(real_pred, valid)

            fake = torch.zeros(input_img.size(0), *self.patch)
            fake = fake.type_as(input_img)
            fake_loss = self.adversarial_loss(fake_pred, fake)

            d_loss = 0.5 * (real_loss + fake_loss) * 10.0

            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            self.log("d_loss", d_loss, prog_bar=True)
            return output

    def validation_step(self, batch, batch_idx):
        # get input and gt batch
        input_img = batch["input"]
        gt_img = batch["gt"]

        # generate image
        with torch.no_grad():
            generated_img = self(input_img)

        # perceptual loss
        gt_img = utils.tensor_to_lpips_format(gt_img)
        generated_img = utils.tensor_to_lpips_format(generated_img)
        if torch.isnan(generated_img).any():
            perceptual_loss = self.val_pre_perceptual_loss
        else:
            # calc lpips @ (batch size, C, H, W)
            perceptual_loss = self.measure.model.forward(gt_img, generated_img).mean()

        # log
        self.log("val_loss", perceptual_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        # get images
        input_img = batch["input"]
        assert input_img.shape[0] == 1, "Test batch size should be 1"
        img_name = os.path.splitext(os.path.basename(batch["input_path"][0]))[0]

        # make dirctries to save result
        paths = ["input", "output"]
        for path in paths:
            utils.mkdir(f"{self.save_dir}/test_results/{path}")

        # adjust image size
        input_original = input_img.detach()
        # h, w = input_img.shape[2], input_img.shape[3]
        # pad_factor = 8
        # pad_r = int((w // pad_factor + 1) * pad_factor - w)
        # pad_b = int((h // pad_factor + 1) * pad_factor - h)
        # # (left, right, top, bottom)
        # p2d = (0, pad_r, 0, pad_b)
        # input_img = F.pad(input_img, p2d, "reflect")

        # generate image
        with torch.no_grad():
            generated_img = self(input_img)
        # generated_img = generated_img[:, :, : h, : w]

        input_np = utils.tensor2img(input_original)
        generated_np = utils.tensor2img(generated_img)

        # save generated image
        filename = f"{img_name}_output.png"
        utils.save_image(f"{self.save_dir}/test_results/output/{filename}", generated_np)

        # save input image
        filename = f"{img_name}_input.png"
        utils.save_image(f"{self.save_dir}/test_results/input/{filename}", input_np)

        # grid image for wandb
        grid_img = utils.make_grid_img([input_original, generated_img], n_rows=1)

        # store results to get all results
        self.test_grids.append(grid_img)

    def on_test_epoch_start(self):
        # make outputs folder inder hydra root
        utils.mkdir(f"{self.save_dir}/test_results")
        # list to save result on each step
        self.test_grids = []

    def on_test_epoch_end(self):
        # upload grid images to wandb
        self.logger.log_image("gbr", self.test_grids)

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
