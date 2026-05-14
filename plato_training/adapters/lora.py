"""
LoRA implementation with save/load round-trip.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Dict, List
from pathlib import Path

try:
    from safetensors.torch import save_file, load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


class LoRALayer(nn.Module):
    def __init__(self, original, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.d_in = original.in_features
        self.d_out = original.out_features
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(self.d_in, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, self.d_out))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.original(x) + self.dropout(x) @ self.lora_A @ self.lora_B * self.scaling

    def merge(self):
        merged = nn.Linear(self.d_in, self.d_out, bias=self.original.bias is not None)
        merged.weight.data = self.original.weight.data + (self.lora_B.T @ self.lora_A.T) * self.scaling
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data.clone()
        return merged

    def num_trainable_params(self):
        return self.lora_A.numel() + self.lora_B.numel()

    def lora_state_dict(self):
        return {"lora_A": self.lora_A.data.clone(), "lora_B": self.lora_B.data.clone()}


def inject_lora(model, rank=8, alpha=16, target_modules=None, dropout=0.0):
    if target_modules is None:
        target_modules = ["W_query", "W_value", "W_key", "W_out"]
    injection_map = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(t in name for t in target_modules):
                lora = LoRALayer(module, rank=rank, alpha=alpha, dropout=dropout)
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], lora)
                injection_map[name] = "lora"
    return injection_map


def save_lora_weights(model, injection_map, path):
    lora_state = {}
    for name in injection_map:
        module = dict(model.named_modules())[name]
        if isinstance(module, LoRALayer):
            lora_state[f"{name}.A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.B"] = module.lora_B.data.cpu()
    if HAS_SAFETENSORS:
        save_file(lora_state, path)
    else:
        torch.save(lora_state, path)
    return Path(path).read_bytes()


def load_lora_weights(model, injection_map, path):
    if HAS_SAFETENSORS:
        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu")
    for name in injection_map:
        module = dict(model.named_modules())[name]
        if isinstance(module, LoRALayer):
            a_key, b_key = f"{name}.A", f"{name}.B"
            if a_key in state and b_key in state:
                module.lora_A.data.copy_(state[a_key])
                module.lora_B.data.copy_(state[b_key])
