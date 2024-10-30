import torch.nn as nn


class GPT2(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x): 
        pass

    def save_to(self, path: str):
        pass

    def load_from(self, path: str):
        pass

