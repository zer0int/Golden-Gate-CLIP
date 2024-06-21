import clip
import torch
from torch import nn as nn
import numpy as np
import os
import json
import warnings

warnings.filterwarnings("ignore")

# Set model and define the text file containing prompts here, and then just run this code.
clipmodel = "ViT-L/14"
text_file = "alltexts.txt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ClipWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_text(x)

def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    model, _ = clip.load(clipmodel, device=device)
    return model.to(device)

def preprocess_text(texts):
    return clip.tokenize(texts, truncate=True).to(device)

def read_texts_from_file(file_path):
    with open(file_path, 'r') as file:
        texts = file.read().strip().split('\n')
    return texts

class LayerCaptureHook:
    def __init__(self, model, layer_idx):
        self.layer_idx = layer_idx
        self.activations = None
        model.transformer.resblocks[layer_idx].mlp.c_fc.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.detach()

    def get_activations(self):
        return self.activations

def register_hooks(model, layers):
    hooks = []
    for layer_idx in layers:
        hooks.append(LayerCaptureHook(model, layer_idx))
    return hooks

def get_top_positive_features(activations, top_k=10):
    positive_indices = [i for i, v in enumerate(activations) if v > 0]
    sorted_indices = sorted(positive_indices, key=lambda i: activations[i], reverse=True)
    return sorted_indices[:top_k]

def find_common_features(top_indices_per_text):
    common_features = set(top_indices_per_text[next(iter(top_indices_per_text))])
    for indices in top_indices_per_text.values():
        common_features.intersection_update(indices)
    return list(common_features)

def store_top_activations_to_json(activations_list, filename="top-TEXT-transformer-activations-ALL.json"):
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(activations_list, f, indent=4)

def store_common_features(common_features_per_layer, filename="top_TEXT-transformer_activations-common.txt"):
    with open(filename, "w", encoding='utf-8') as f:
        f.write("Identified Common Features\n")
        for layer, features in common_features_per_layer.items():
            features_str = ", ".join(map(str, features))
            f.write(f"Layer {layer}: {features_str}\n")

# Main processing
model = load_clip_model()
texts = read_texts_from_file(text_file)
text_inputs = preprocess_text(texts)

layers = [8, 9, 10, 11]
hooks = register_hooks(model, layers)

all_top_neurons_per_text = {layer: {} for layer in layers}

for text_input, text_str in zip(text_inputs, texts):
    with torch.no_grad():
        _ = model.encode_text(text_input.unsqueeze(0))
    for hook in hooks:
        activations = hook.get_activations()
        if activations is not None:
            cls_activations = activations[:, 0, :].cpu().tolist()
            top_positive_features = [get_top_positive_features(act, top_k=10) for act in cls_activations]
            all_top_neurons_per_text[hook.layer_idx][text_str] = top_positive_features

store_top_activations_to_json(all_top_neurons_per_text, "top-TEXT-transformer-activations-ALL.json")

common_features_per_layer = {}
for layer, activations_per_text in all_top_neurons_per_text.items():
    flattened_activations = {k: [item for sublist in v for item in sublist] for k, v in activations_per_text.items()}
    common_features = find_common_features(flattened_activations)
    common_features_per_layer[layer] = common_features

store_common_features(common_features_per_layer, "top_TEXT-transformer_activations-common.txt")

print("Processing complete. Results saved to top-TEXT-transformer-activations-ALL.json and top_TEXT-transformer_activations-common.txt.")
