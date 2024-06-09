import clip
from clip.model import QuickGELU
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
import torchvision.utils
import numpy as np
import random
import pdb
import collections
from typing import Any
import argparse
from argparse import Namespace
from PIL import Image
import json
import csv
import os
import sys
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
# Custom imports
from image_net import TotalVariation, CrossEntropyLoss, MatchBatchNorm, BaseFakeBN, LayerActivationNorm
from image_net import ActivationNorm, NormalVariation, ColorVariation, fix_random_seed
from image_net import NetworkPass
from image_net import LossArray, TotalVariation
from image_net import ViTFeatHook, ViTEnsFeatHook
from regularizers import TotalVariation as BaseTotalVariation, FakeColorDistribution as AbstractColorDistribution
from regularizers import FakeBatchNorm as BaseFakeBN, NormalVariation as BaseNormalVariation
from regularizers import ColorVariation as BaseColorVariation
from hooks import ViTAttHookHolder, ViTGeLUHook, ClipGeLUHook, SpecialSaliencyClipGeLUHook
from prepost import Clip, Tile, Jitter, RepeatBatch, ColorJitter, fix_random_seed
from prepost import GaussianNoise
from util import ClipWrapper
from util import new_init, save_intermediate_step, save_image, fix_random_seed

# Set model and define a folder containing images here, and then just run this code.
clipmodel = "ViT-L/14"
image_folder = "goldengate"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_clip_dimensions(clipmodel):
    model, preprocess = clip.load(clipmodel)
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, transforms.Resize):
            input_dims = transform.size
            break
    num_layers = None
    num_features = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features
    return input_dims, num_layers, num_features

def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    model, _ = clip.load(clipmodel, device=device)
    model = ClipWrapper(model).to(device)
    return model

def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return list(map(int, range_str.split(',')))

clipname = clipmodel.replace("/", "-").replace("@", "-")
model = load_clip_model()

input_dims, num_layers, num_features = get_clip_dimensions(clipmodel)

transforming = transforms.Compose([
    transforms.Resize((input_dims, input_dims)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

print(f"Selected input dimension for {clipmodel}: {input_dims}")
print(f"Number of Layers: {num_layers} with {num_features} Features / Layer\n")



from clip.model import ResidualAttentionBlock

class ClipAttentionCaptureHook:
    def __init__(self, module: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.attn_scores = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # Capture the attention scores, assuming that output[1] exists
        self.attn_scores = output[1].detach() if isinstance(output, tuple) and len(output) > 1 else None
        #print("self.attn_scores:\n", self.attn_scores, "\n")

    def get_top_attention(self, k=10):
        if self.attn_scores is not None:
            top_values, top_indices = torch.topk(self.attn_scores, k, dim=-1)
            return self.layer_idx, top_values, top_indices
        return None, None, None

# Function to register hooks across all attention layers
def register_attention_hooks(model, num_layers):
    hooks = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            hook = ClipAttentionCaptureHook(module, layer_idx)
            hooks.append(hook)
            layer_idx += 1
            if layer_idx >= num_layers:
                break
    return hooks

# Modify the attention method in ResidualAttentionBlock to return attention weights
def modified_attention(self, x: torch.Tensor):
    self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
    attn_output, attn_output_weights = self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)
    return attn_output  # Return only the attention output tensor

# Patch the ResidualAttentionBlock's attention method
for module in model.modules():
    if isinstance(module, ResidualAttentionBlock):
        module.attention = modified_attention.__get__(module, ResidualAttentionBlock)

# After the forward pass
def get_all_top_attention(hooks, k=10):
    all_top_attention = []
    for hook in hooks:
        layer_idx, top_values, top_indices = hook.get_top_attention(k)
        if top_values is not None:
            all_top_attention.append((layer_idx, top_values, top_indices))
    return all_top_attention

# Register hooks for all attention layers
attention_hooks = register_attention_hooks(model, num_layers)

# Process all images in the "goldengate" folder for attention
all_top_attention_per_image = {}

for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    if os.path.isfile(image_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        img = Image.open(image_path)
        input_image = transforming(img).unsqueeze(0).to(device)

        # Perform your forward pass with the input image
        output = model(input_image)

        # Retrieve top attention scores across all layers
        all_top_attention = get_all_top_attention(attention_hooks, k=10)

        #print(f"\nProcessing {image_name} for attention layers:\n")
        #for layer_idx, top_values, top_indices in all_top_attention:
        #    print(f"Layer {layer_idx}, Top Values: {top_values}, Indices: {top_indices}")

        top_attention_per_layer = {}

        for layer_idx, top_values, top_indices in all_top_attention:
            # Convert tensor to a list of integers for each instance in the batch
            attention_indices = top_indices[0][0].cpu().tolist()
            top_attention_per_layer[layer_idx] = attention_indices

        all_top_attention_per_image[image_name] = top_attention_per_layer

# Save results for all images to a JSON file
with open(f"top-ATTN-activations-ALL.json", "w", encoding='utf-8') as f:
    json.dump(all_top_attention_per_image, f, indent=4)

print("\nProcessing complete. Attention activations saved to top-attn-activations-all-test.json.")

# Find common attention features
def find_common_features(all_top_attention_per_image):
    layer_keys = list(next(iter(all_top_attention_per_image.values())).keys())
    common_features_per_layer = {layer: set() for layer in layer_keys}

    for layer in layer_keys:
        all_feature_sets = [set(image_data[layer]) for image_data in all_top_attention_per_image.values()]
        common_features = set.intersection(*all_feature_sets)
        common_features_per_layer[layer] = list(common_features)

    return common_features_per_layer

common_attention_features_per_layer = find_common_features(all_top_attention_per_image)

# Find common attention features
common_attention_features_per_layer = find_common_features(all_top_attention_per_image)

# Save common attention features to a text file
with open(f"top_ATTN_activations-common.txt", "w", encoding='utf-8') as f:
    f.write("Identified Common Attention Features\n")
    for layer, features in common_attention_features_per_layer.items():
        features_str = ", ".join(map(str, features))
        f.write(f"Layer {layer}: {features_str}\n")

print("Common attention features also saved to top_ATTN_activations-common.txt.")