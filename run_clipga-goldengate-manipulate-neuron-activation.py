import clip
import imageio
import torchvision
import PIL.Image
#from IPython import display
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
#checkin_step = training_iterations - 1
checkin_step = 10
import os
import sys
import kornia
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import Resize
import random
import numpy as np
import argparse
import glob
from multiprocessing import cpu_count
from tqdm import tqdm
import warnings
import pickle
import warnings
from colorama import Fore, Style
import pdb
warnings.filterwarnings('ignore')

from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

clipmodel = 'ViT-L/14' 
training_iterations = 300 # 200-400 for good "opinion" / loss.
batch_size = 12 # CUDA OOM? Try lowering this.
many_tokens = 4 # 4-6 is best; much more will be arbitrary.
input_dims = 224 # 336 for ViT-L/14@336px

# Now scroll ALL THE WAY DOWN to loop()!


parser = argparse.ArgumentParser(description="CLIP Gradient Ascent")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image, e.g.: --image_path goldengate/goldengate1.png")
args = parser.parse_args()

perceptor, preprocess = clip.load(clipmodel, jit=False)
perceptor = perceptor.eval().float()

def displ(img, pre_scaled=True):
    img = np.array(img)[:,:,:]
    img = np.transpose(img, (1, 2, 0))
    if not pre_scaled:
        img = scale(img, 48*4, 32*4)
    imageio.imwrite(str(3) + '.png', np.array(img))
    return display.Image(str(3)+'.png')

def clip_encode_text(gobble, text):
    x = torch.matmul(text, gobble.token_embedding.weight)  # [batch_size, n_ctx, d_model]
    x = x + gobble.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = gobble.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = gobble.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ gobble.text_projection
    return x

prompt = clip.tokenize('''''').numpy().tolist()[0]
prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

img_path = args.image_path
img_name = os.path.splitext(os.path.basename(img_path))[0]

im = torch.tensor(imageio.imread(img_path).copy()).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
im = F.interpolate(im, (input_dims, input_dims))

torch.cuda.empty_cache()

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000
        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1
        ptt = prompt
        self.prompt = torch.zeros(batch_size, len(ptt), 49408).cuda()
        for jk, pt in enumerate(ptt):
            self.prompt[:, jk, pt] = 1
        self.pad = torch.zeros(batch_size, 77 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
        fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
        return fin

lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{'params': mapper, 'lr': 5}])

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
).cuda()

tok = clip.simple_tokenizer.SimpleTokenizer()

bests = {1000:'None', 1001:'None', 1002:'None', 1003:'None', 1004:'None'}

def augment(into):
    into = augs(into)
    return into

# Custom hook to scale the feature activation
class FeatureScalerHook:
    def __init__(self, model, layer_idx, feature_idx, scale_factor):
        self.model = model
        self.layer_idx = layer_idx
        self.feature_idx = feature_idx
        self.scale_factor = scale_factor
        self.handle = None
        self.register_hook()

    def register_hook(self):
        def hook(module, input, output):
            output[:, :, self.feature_idx] *= self.scale_factor
            return output

        layer = self.model.visual.transformer.resblocks[self.layer_idx].mlp.c_fc
        self.handle = layer.register_forward_hook(hook)

    def remove(self):
        if self.handle:
            self.handle.remove()

def ascend_txt():
    global im
    iii = nom(augment(im[:,:3,:,:].expand(64, -1, -1, -1)))
    iii = perceptor.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(perceptor, lll)
    return -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, batch_size).T.mean(1), lll

def train():
    with autocast():
        loss1, lll = ascend_txt()
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, lll


def checkin(loss, lll):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}")
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('!', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace(')', '')
        j = j.replace('(', '')
        j = j.replace('*', '')
        j = j.replace(',', '')

        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"TOK/tokens_{img_name}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# ========================= Top neurons found for "goldengate" images: =========================
# Layer 22, Feature 2435: The SAN FRANCISCO, sf, bayarea, etc. neuron!
# Layer 22, Feature 870: California and travel neuron (not just sf)
#-----
# Layer 22, Feature 2431: "historical famous government thingy" ~CLIP.
# Layer 22, Feature 2616: Bridge, but many arbitrary things as well.
# Layer 22, Feature 1108: Travel, Immigration, Americans, Huge American Things, Canada / GB / USA / LatAm
# Layer 22, Feature 2432: "indeed shortly yet followed more already there quietly again", weird neuron of adverbs!
# Layer 22, Feature 2432: Neuron of MORE. BIGGER. Truth.
# Layer 22, Feature 3366: Neuron of ooooooooh, !!!!!!!!!!!, and amazement.
#-----
# Layer 21, Feature 1844: Excited joy neuron of woooohoooo!! :-D
# Layer 21, Feature 2373: Similar holiday and joy neuron, but more slangy? xD
# Layer 21, Feature 490: Flying, drones, and aerial photography neuron
# Layer 21, Feature 3483: Seems to feed into "adverb neuron", similar output when scale_factor=1000

def loop():
    scaling_factors = [10] # can be e.g. [10, 100, 1000] do for all -> scale_factor=factor
    top_activations_layer_21 = [1844]
    top_activations_layer_22 = [2435]#[2435]#[1108]#[2432]#[1108]#[1108, 2432]
    for factor in scaling_factors:
        print(f"\nTesting with scaling factor: {factor}\n")
        
        # Scale activations in layer 21
        hooks_layer_21 = []
        for feature_idx in top_activations_layer_21:
            hook = FeatureScalerHook(perceptor, layer_idx=21, feature_idx=feature_idx, scale_factor=1)# 1 = normal, 100 = nice balanced effect, 10 = subtle / unnoticable
            hooks_layer_21.append(hook)
            
        # Scale activations in layer 22
        hooks_layer_22 = []
        for feature_idx in top_activations_layer_22:
            hook = FeatureScalerHook(perceptor, layer_idx=22, feature_idx=feature_idx, scale_factor=1000)# 1000 = crank it up! ~10000 and gradients explode. That's a NaN.
            hooks_layer_22.append(hook)
        
        for i in range(training_iterations):
            loss, lll = train()
            if i % checkin_step == 0:
                checkin(loss, lll)
                #print(Fore.YELLOW + f"Iteration {i}: Average Loss: {loss.mean().item()}")
loop()