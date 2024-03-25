import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
import folder_paths
from models.baseline import BiRefNet
from config import Config

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms

from loguru import logger
from folder_paths import models_dir

config = Config()


class BiRefNet_img_processor:
    def __init__(self, config):
        self.config = config
        self.data_size = (config.size, config.size)
        self.transform_image = transforms.Compose([
            transforms.Resize(self.data_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __call__(self, _image: np.array):
        _image_rs = cv2.resize(_image, (self.config.size, self.config.size), interpolation=cv2.INTER_LINEAR)
        _image_rs = Image.fromarray(np.uint8(_image_rs*255)).convert('RGB')
        image = self.transform_image(_image_rs)
        return image



class BiRefNet_node:
    def __init__(self):
        self.ready = False

    def load(self, weight_path, device, verbose=False):
        # load model
        self.model = BiRefNet()
        state_dict = torch.load(weight_path, map_location='cpu')
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(device)
        self.model.eval()

        # load processor
        self.processor = BiRefNet_img_processor(config)

        self.ready = True
        if verbose:
            logger.debug("BiRefNet load finish")


    @classmethod
    def INPUT_TYPES(s):
        if torch.backends.mps.is_available():
            mps_device = ["mps"]
        else:
            mps_device = []
        
        if torch.cuda.is_available():
            cuda_deviceds = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            cuda_deviceds = []

        return {
            "required": {
                "device": (["auto", "cpu"] + cuda_deviceds + mps_device, {"default": "auto"}),
                "image": ("IMAGE", {}),
            }
        }


    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = "matting"
    CATEGORY = "Fooocus"

    def matting(self, image, device):
        # process auto deivce
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if not self.ready:
            weight_path = os.path.join(models_dir, "BiRefNet", "BiRefNet-ep480.pth")
            self.load(weight_path, device=device)
        
        image = image.squeeze().cpu().numpy()
        img = self.processor(image)
        inputs = img[None, ...].to(device)
        logger.debug(f"{inputs.shape}")
        
        with torch.no_grad():
            scaled_preds = self.model(inputs)[-1].sigmoid()

        res = nn.functional.interpolate(
            scaled_preds[0].unsqueeze(0),
            size=image.shape[:2],
            mode='bilinear',
            align_corners=True
            )
        return res



NODE_CLASS_MAPPINGS = {
    "BiRefNet": BiRefNet_node,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet": "BiRefNet Segmentation",
}

