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
    num_layers = len(model.visual.transformer.resblocks)
    num_features = model.visual.transformer.resblocks[-1].mlp.c_fc
    return input_dims, num_layers, num_features

def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    model, _ = clip.load(clipmodel, device=device)
    model = ClipWrapper(model).to(device)
    return model

model = load_clip_model()

input_dims, num_layers, num_features = get_clip_dimensions(clipmodel)
print(f"Selected input dimension for {clipmodel}: {input_dims}")
print(f"Number of Layers: {num_layers} with {num_features} Features / Layer\n")

transforming = transforms.Compose([
    transforms.Resize((input_dims, input_dims)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x[:3, :, :]),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

class ClipNeuronCaptureHook:
    def __init__(self, module: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.detach()[:, 0, :]  # Extract CLS token activations

    def get_top_activations(self, k=10):
        if self.activations is not None:
            top_values, top_indices = torch.topk(self.activations, k, dim=-1)
            return top_values.cpu().numpy(), top_indices.cpu().numpy()
        return None, None

def register_hooks(model, layers_to_capture):
    hooks = []
    for layer_idx in layers_to_capture:
        module = model.clip.visual.transformer.resblocks[layer_idx].mlp.c_fc
        hook = ClipNeuronCaptureHook(module, layer_idx)
        hooks.append(hook)
    return hooks

def get_all_top_activations(hooks, k=10):
    all_top_activations = []
    for hook in hooks:
        top_values, top_indices = hook.get_top_activations(k)
        if top_values is not None:
            all_top_activations.append((hook.layer_idx, top_values, top_indices))
    return all_top_activations

image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith(('png', 'jpg', 'jpeg'))]

def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return list(map(int, range_str.split(',')))


layers_to_capture = list(range(1, 24))
hooks = register_hooks(model, layers_to_capture)

all_activations_per_image = []

for image_path in image_paths:
    img = Image.open(image_path).convert('RGB')
    input_image = transforming(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(input_image)
    all_activations = get_all_top_activations(hooks)
    all_activations_per_image.append((os.path.basename(image_path), all_activations))

def store_top_activations_to_csv(activations_list, filename="top_preGELU-ONLY-CLS_activations.csv"):
    # Prepare a dictionary to hold the activations per layer
    activations_per_layer = {layer: [] for layer in layers_to_capture}
    image_filenames = []

    for img_filename, img_activations in activations_list:
        image_filenames.append(img_filename)
        for idx, values, indices in img_activations:
            activations_per_layer[idx].append(indices[0].tolist())  # Extract the CLS token top activations

    # Ensure all columns have the same length by filling shorter columns with empty strings
    max_length = max(len(img_indices) for indices_list in activations_per_layer.values() for img_indices in indices_list)
    for layer in layers_to_capture:
        for indices_list in activations_per_layer[layer]:
            while len(indices_list) < max_length:
                indices_list.append("")

    # Transpose the lists to have layers as columns
    transposed_indices = list(zip(*[activations_per_layer[layer] for layer in layers_to_capture]))

    # Write the activations to a CSV file
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Filename"] + [f"Layer {layer}" for layer in layers_to_capture])

        for i, row in enumerate(transposed_indices):
            writer.writerow([image_filenames[i]] + [",".join(map(str, indices_list)) for indices_list in row])

# Store top activations in a CSV file
store_top_activations_to_csv(all_activations_per_image)

# Identify common features across all images for each layer
def identify_common_features(activations_list):
    common_features_per_layer = {layer: set() for layer in layers_to_capture}

    for idx, (img_filename, img_activations) in enumerate(activations_list):
        if idx == 0:  # Initialize the common features with the first image's activations
            for layer_idx, _, indices in img_activations:
                common_features_per_layer[layer_idx] = set(indices[0].tolist())
        else:  # Intersect with the subsequent images' activations
            for layer_idx, _, indices in img_activations:
                common_features_per_layer[layer_idx] &= set(indices[0].tolist())

    return common_features_per_layer

# Store common features to a text file
def store_common_features(common_features, filename="top_preGELU-ONLY-CLS_activations-common.txt"):
    with open(filename, 'w') as file:
        file.write("Identified Common Features\n")
        for layer, features in common_features.items():
            file.write(f"Layer {layer}: {', '.join(map(str, features))}\n")

# Identify and store common features
common_features = identify_common_features(all_activations_per_image)
store_common_features(common_features)

print("Top activations for CLS stored successfully in top_preGELU-ONLY-CLS_activations.csv")
print("Common features stored successfully in top_preGELU-ONLY-CLS_activations-common.txt")
