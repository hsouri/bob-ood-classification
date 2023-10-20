from midas.model_loader import default_models, load_model
from midas.dpt_depth import DPT, DPTDepthModel

from functools import partial

import torch
import torch.nn as nn

from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

# model_path = "/fs/cml-projects/benchmarking_backbone/checkpoints/dpt_swin2_tiny_256.pt"
model_path = "/srv/share4/virajp/backbone_benchmark/dpt_swin2_tiny_256.pt"
model_urls = {
     "swin2_tiny_256_midas": '/srv/share4/virajp/backbone_benchmark/dpt_swin2_tiny_256.pt',
     "dpt_swin2_base_384": '/fs/cml-projects/benchmarking_backbone/checkpoints/dpt_swin2_base_384.pt'
    }

import pdb

@register_model
def swin2_tiny_256_midas(pretrained=False, **kwargs):
    midas_model = DPTDepthModel(
        path=model_urls['swin2_tiny_256_midas'],
        backbone="swin2t16_256",
        non_negative=True,
    )
    num_features = 768
    model = midas_model.pretrained.model
    return model

@register_model
def swin2_base_384_midas(pretrained=False, **kwargs):
    midas_model = DPTDepthModel(
        path=model_urls['dpt_swin2_base_384'],
        backbone="swin2b24_384",
        non_negative=True,
    )
    num_features = 768
    model = midas_model.pretrained.model
    return model

