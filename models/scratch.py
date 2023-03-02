import importlib
from abc import *
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from models.small import SmallModel

class SmallModel(nn.Module, ABC):
    def __init__(self, small_model:str, small_config:dict) -> None:
        super().__init__()

        model = importlib.import_module('models.small').__getattribute__(small_model)
        self.small_model:SmallModel =  model(**small_config)

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return self.small_model(*args, **kwargs)
    

