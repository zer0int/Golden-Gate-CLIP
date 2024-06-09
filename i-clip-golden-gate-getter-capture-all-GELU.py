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
        if isinstance(transform, Resize):
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

class ClipNeuronCaptureHook:
    def __init__(self, module: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = None
        self.top_values = None
        self.top_indices = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.detach()


    def get_top_neurons(self, k=10):
        if self.activations is not None:
            self.top_values, self.top_indices = torch.topk(self.activations, k, dim=-1)
            return self.layer_idx, self.top_values, self.top_indices
        return None, None, None


def register_hooks(model, num_layers):
    hooks = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, QuickGELU):
            hook = ClipNeuronCaptureHook(module, layer_idx)
            hooks.append(hook)
            layer_idx += 1
            if layer_idx >= num_layers:
                break
    return hooks

def get_all_top_neurons(hooks, k=10):
    all_top_neurons = []
    for hook in hooks:
        layer_idx, top_values, top_indices = hook.get_top_neurons(k)
        if top_values is not None:
            all_top_neurons.append((layer_idx, top_values, top_indices))
    return all_top_neurons

hooks = register_hooks(model, num_layers)

all_top_neurons_per_image = {}

for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    if os.path.isfile(image_path):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        img = Image.open(image_path)
        input_image = transforming(img).unsqueeze(0).to(device)

        output = model(input_image)

        all_top_neurons = get_all_top_neurons(hooks, k=10)

        #print(f"\nProcessing {image_name}:\n")
        #for layer_idx, top_values, top_indices in all_top_neurons:
        #    print(f"Layer {layer_idx}, Top Values: {top_values}, Indices: {top_indices}")

        top_features_per_layer = {}

        for layer_idx, top_values, top_indices in all_top_neurons:
            # Convert tensor to a list of integers for each instance in the batch
            feature_indices = top_indices[0][0].cpu().tolist()  # Converts the first instance to a CPU tensor, then to a list
            top_features_per_layer[layer_idx] = feature_indices

        all_top_neurons_per_image[image_name] = top_features_per_layer

with open(f"top-GELU-activations-ALL.json", "w", encoding='utf-8') as f:
    json.dump(all_top_neurons_per_image, f, indent=4)

print("Processing complete. Results saved to top-GELU-activations-ALL.json.")

# Additional processing to find common features
def find_common_features(all_top_neurons_per_image):
    layer_keys = list(next(iter(all_top_neurons_per_image.values())).keys())
    common_features_per_layer = {layer: set() for layer in layer_keys}

    for layer in layer_keys:
        all_feature_sets = [set(image_data[layer]) for image_data in all_top_neurons_per_image.values()]
        common_features = set.intersection(*all_feature_sets)
        common_features_per_layer[layer] = list(common_features)

    return common_features_per_layer

common_features_per_layer = find_common_features(all_top_neurons_per_image)

# May need this for something else later

#with open(f"top_GELU_activations-common.json", "w", encoding='utf-8') as f:
#    json.dump(common_features_per_layer, f, indent=4)

#print("Common features saved to top_GELU_activations-common.json.")

# Save common features to a text file
with open(f"top_GELU_activations-common.txt", "w", encoding='utf-8') as f:
    f.write("Identified Common Features\n")
    for layer, features in common_features_per_layer.items():
        features_str = ", ".join(map(str, features))
        f.write(f"Layer {layer}: {features_str}\n")

print("Common features saved to top_GELU_activations-common.txt.")

