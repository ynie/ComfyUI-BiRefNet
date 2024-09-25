import os
import sys
from typing import Tuple

import math

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
from image_proc import refine_foreground

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
                "input_image": ("IMAGE", {}),
            },
            "optional": {
                "apply_fast_foreground_estimation": ("BOOLEAN", {"default": True},),
                "model_name": ("STRING", {"default": "BiRefNet-general-epoch_244.pth", "multiline": False, "dynamicPrompts": False}),
                "reference_image": ("IMAGE", {}),
                "reference_image_padding": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", )
    RETURN_NAMES = ("mask", )
    FUNCTION = "matting"
    CATEGORY = "Fooocus"

    def matting(self,
                input_image,
                apply_fast_foreground_estimation=True,
                model_name: str = "BiRefNet-general-epoch_244.pth",
                reference_image=None,
                reference_image_padding=0):
        reference_pil_image = None
        leading = 0
        top = 0
        trailing = 0
        bottom = 0
        if reference_image is not None:
            input_pil_image = tensor2pil(input_image)
            reference_pil_image = tensor2pil(reference_image)
            assert input_pil_image.size == reference_pil_image.size

            input_image_width, input_image_height = input_pil_image.size
            leading, top, trailing, bottom = reference_pil_image.getbbox()
            leading = max(0, leading - reference_image_padding)
            top = max(0, top - reference_image_padding)
            trailing = min(input_image_width, trailing + reference_image_padding)
            bottom = min(input_image_height, bottom + reference_image_padding)

            content_image = input_pil_image.crop((leading, top, trailing, bottom))
            content_image.save("modified_input.png")

            input_image = pil2tensor(content_image)

        image_masked = self._process(input_image, apply_fast_foreground_estimation, model_name)
        if reference_pil_image is not None:
            merged_image = Image.new("RGBA", reference_pil_image.size)
            merged_image.paste(image_masked, box=(leading, top, trailing, bottom), mask=image_masked)
            image_masked = merged_image

        mask = np.array(image_masked.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)
        return mask,

    def _process(self, input_image, apply_fast_foreground_estimation, model_name: str):
        # process auto device
        device = comfy.model_management.get_torch_device()

        global BI_REF_NET_MODEL
        global BI_REF_NET_PROCESSOR

        if BI_REF_NET_MODEL is None:
            model = BiRefNet()
            weight_path = os.path.join(models_dir, "BiRefNet", model_name)
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

        img = BI_REF_NET_PROCESSOR(input_image.squeeze().numpy())
        inputs = img[None, ...].to(device)
        logger.debug(f"{inputs.shape}")

        with torch.no_grad():
            preds = BI_REF_NET_MODEL(inputs)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()

        input_pil_image = tensor2pil(input_image)

        # Show Results
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(input_pil_image.size)

        if apply_fast_foreground_estimation:
            image_masked = refine_foreground(input_pil_image, pred_pil)
            image_masked.putalpha(pred_pil.resize(input_pil_image.size))
            return image_masked
        else:
            input_pil_image.putalpha(pred_pil.resize(input_pil_image.size))
            return input_pil_image


def pil2mask(image):
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def aspect_fit_size(image_size: Tuple[int, int], canvas_size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = image_size
    max_width, max_height = canvas_size

    if width > max_width or height > max_height:
        result_width, result_height = canvas_size
        mw = float(max_width) / float(width)
        mh = float(max_height) / float(height)

        if mh < mw:
            result_width = max_height / height * width
        elif mw < mh:
            result_height = max_width / width * height
    else:
        result_width, result_height = image_size
    return int(math.floor(result_width)), int(math.floor(result_height))


NODE_CLASS_MAPPINGS = {
    "BiRefNet": BiRefNet_node,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet": "BiRefNet Segmentation",
}
