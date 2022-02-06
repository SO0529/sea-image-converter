import torch
from pytorch_lightning.callbacks import Callback
from utils import utils


class ValImageLogger(Callback):
    def __init__(self, val_dataloader):
        super().__init__()
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer, pl_module):
        # get first batch from val_dataloader
        val_imgs = iter(self.val_dataloader).__next__()
        input_img = val_imgs["input"].to(pl_module.device)
        gt_img = val_imgs["gt"].to(pl_module.device)

        with torch.no_grad():
            generated_img = pl_module(input_img)

        grid_imgs = []
        for batch_idx in range(input_img.shape[0]):
            grid_imgs.append(utils.make_grid_img(
                [input_img[batch_idx], gt_img[batch_idx], generated_img[batch_idx]],
                n_rows=1
                ))

        trainer.logger.log_image("val_results", grid_imgs)
