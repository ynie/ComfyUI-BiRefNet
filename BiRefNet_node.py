import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
import folder_paths
from models.birefnet import BiRefNet
from config import Config
import comfy.model_management

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

BI_REF_NET_MODEL = None
BI_REF_NET_PROCESSOR = None

class BiRefNet_node:
    # Correctly move INPUT_TYPES to the class level
    @classmethod
    def INPUT_TYPES(cls):
        # Example structure, adjust according to your actual input requirements
        return {
            "required": {
                "image": ("IMAGE", {}),
            },
            "optional": {
                # Define optional inputs if any
            }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = "matting"
    CATEGORY = "Fooocus"

    def matting(self, image):
        # process auto device
        device = comfy.model_management.get_torch_device()

        global BI_REF_NET_MODEL
        global BI_REF_NET_PROCESSOR

        if BI_REF_NET_MODEL is None:
            model = BiRefNet()
            weight_path = os.path.join(models_dir, "BiRefNet", "BiRefNet_DIS_ep580.pth")
            state_dict = torch.load(weight_path, map_location=device)
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            BI_REF_NET_MODEL = model

        if BI_REF_NET_PROCESSOR is None:
            BI_REF_NET_PROCESSOR = BiRefNet_img_processor(config)

        image = image.squeeze().numpy()
        img = BI_REF_NET_PROCESSOR(image)
        inputs = img[None, ...].to(device)
        logger.debug(f"{inputs.shape}")
        
        with torch.no_grad():
            BI_REF_NET_MODEL.to(device)  # Move the model to the selected device
            scaled_preds = BI_REF_NET_MODEL(inputs)[-1].sigmoid()

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
