# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
import math
from typing import Optional, List, Type, Set, Literal

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from datetime import datetime

UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
#     "Transformer2DModel",  # どうやらこっちの方らしい？ # attn1, 2
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "DownBlock2D",
    "UpBlock2D",
    
]  # locon, 3clier

LORA_PREFIX_UNET = "lora_unet"

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "xattn-up", # all up blocks only
    "xattn-down",# all down blocks only
    "xattn-mid",# mid blocks only
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]

def load_ortho_dict(n):
    path = f'/sensei-fs/users/rgandikota/orthogonal_basis/{n:09}.ckpt'
    if os.path.isfile(path):
        return torch.load(path)
    else:
        x = torch.randn(n,n)
        eig, _, _ = torch.svd(x)
        torch.save(eig, path)
        return eig

def init_ortho_proj(rank, weight):
    seed = torch.seed()
    torch.manual_seed(datetime.now().timestamp())
    q_index = torch.randint(high=weight.size(0),size=(rank,))
    torch.manual_seed(seed)

    ortho_q_init = load_ortho_dict(weight.size(0)).to(dtype=weight.dtype)[:,q_index]
    return nn.Parameter(ortho_q_init)


class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        train_method='xattn'
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in org_module.__class__.__name__:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif "Conv" in org_module.__class__.__name__:  # 一応
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}")

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        if train_method == 'full':
            nn.init.zeros_(self.lora_up.weight)
        else:
            self.lora_up.weight = init_ortho_proj(lora_dim, self.lora_up.weight)
            self.lora_up.weight.requires_grad_(False)
        
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )


class LoRANetwork(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        train_method: TRAINING_METHODS = "full",
        layers = ['Linear', 'Conv']
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.multiplier = multiplier
        self.lora_dim = rank
        self.alpha = alpha
        self.train_method=train_method
        # LoRAのみ
        self.module = LoRAModule

        # unetのloraを作る
        self.unet_loras = self.create_modules(
            LORA_PREFIX_UNET,
            unet,
            DEFAULT_TARGET_REPLACE,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
            layers = layers
        )
        print(f"create LoRA for U-Net: {len(self.unet_loras)} modules.")

        # assertion 名前の被りがないか確認しているようだ
        lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        # 適用する
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(
                lora.lora_name,
                lora,
            )

        del unet

        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
        layers: List[str],
    ) -> list:
        filt_layers = []
        if 'Linear' in layers:
            filt_layers.extend(["Linear", "LoRACompatibleLinear"])
        if 'Conv' in layers:
            filt_layers.extend(["Conv2d", "LoRACompatibleConv"])
        loras = []
        names = []
        for name, module in root_module.named_modules():
            if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":  # Cross Attention と Time Embed 以外学習
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":  # Cross Attention 以外学習
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":  # Self Attention のみ学習
                if "attn1" not in name:
                    continue
            elif train_method in ["xattn", "xattn-strict", "xattn-up", "xattn-down", "xattn-mid"]:  # Cross Attention
                if "attn" not in name:
                    continue
                if train_method == 'xattn-up':
                    if 'up_block' not in name:
                        continue
                if train_method == 'xattn-down':
                    if 'down_block' not in name:
                        continue
                if train_method == 'xattn-mid':
                    if 'mid_block' not in name:
                        continue
            elif train_method == "full":  # 全部学習
                pass
            else:
                raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in filt_layers:
                        
                            
                        if train_method == 'xattn-strict':
                            if 'out' in child_name:
                                continue
                            if 'to_q' in child_name:
                                continue
                        if train_method == 'noxattn-hspace':
                            if 'mid_block' not in name:
                                continue
                        if train_method == 'noxattn-hspace-last':
                            if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                                continue
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
#                         print(f"{lora_name}")
                        lora = self.module(
                            lora_name, child_module, multiplier, rank, self.alpha, train_method
                        )
#                         print(name, child_name)
#                         print(child_module.weight.shape)
                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)
        # print(f'@@@@@@@@@@@@@@@@@@@@@@@@@@@@ \n {names}')
        return loras

    def prepare_optimizer_params(self):
        all_params = []

        if self.unet_loras:  # 実質これしかない
            params = []
            if self.train_method == 'full':
                [params.extend(lora.parameters()) for lora in self.unet_loras]
            else:
                [params.extend(lora.lora_down.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params

    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
