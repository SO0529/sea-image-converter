import os
import io
import warnings
from omegaconf import OmegaConf
from google.cloud import storage
from google.cloud.storage import Blob
from PIL import Image
import numpy as np
import torch

from nets.sea_image_conveter import SeaImageConverter
from utils import utils

warnings.filterwarnings("ignore")
CONFIG_FILE = "./config/predict.yaml"
DEBUG = False


def predict() -> None:
    # laod config
    cfg = OmegaConf.load(CONFIG_FILE)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # cread model
    model = SeaImageConverter(cfg)
    model.load_state_dict(torch.load(cfg.model_path))
    model.eval().to(device)

    # gcs
    client_storage = storage.Client()
    bucket = client_storage.get_bucket(cfg.bucket_name)

    datasets = [dataset for dataset in cfg.target_dir.split(',')] 

    for dataset in datasets:
        dataset_name = dataset.split("/")[-1]
        files = [file.name for file in client_storage.list_blobs(bucket, prefix=dataset)]
        for file in files:
            torch.cuda.empty_cache()
            img_name = os.path.splitext(os.path.basename(file))[0]

            # get image from gcs
            blob = bucket.blob(file)
            input_img = Image.open(io.BytesIO(blob.download_as_string()))
            input_img = np.array(input_img)
            input_img = utils.uint2single(input_img)
            input_img = utils.single2tensor3(input_img)
            input_img = input_img[None, :, :, :]
            input_img = input_img.to(device)
            with torch.no_grad():
                generated_img = model(input_img)
            generated_np = utils.tensor2img(generated_img)

            # save generated image
            generated_pil = Image.fromarray(np.uint8(generated_np))

            # upload generated image
            bio = io.BytesIO()
            generated_pil.save(bio, format='png')
            blob = Blob(f"{cfg.save_dir}/{dataset_name}/{img_name}.png", bucket)
            blob.upload_from_string(data=bio.getvalue(), content_type="image/png")

            # make grid image
            grid_img = utils.make_grid_img([input_img, generated_img], n_rows=1)

            # save generated image
            grid_np = utils.tensor2img(grid_img)
            grid_pil = Image.fromarray(np.uint8(grid_np))

            # upload grid image
            bio = io.BytesIO()
            grid_pil.save(bio, format='png')
            blob = Blob(f"{cfg.concat_save_dir}/{dataset_name}/{img_name}_concat.png", bucket)
            blob.upload_from_string(data=bio.getvalue(), content_type="image/png")



if __name__ == "__main__":
    predict()
