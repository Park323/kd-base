from typing import Tuple, List, Union

import torch
import torch.nn as nn


class Extractor(nn.Module):
    def __init__(self)->None:
        super().__init__()

    def extract(self, inputs)->torch.Tensor:
        pass

def load_extractor(ext_type=None):
    if ext_type == '':
        return None