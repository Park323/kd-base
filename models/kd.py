import importlib
from abc import *
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .small import SmallModel
from .cumbersome import CumbersomeModel


class AbsKnowledgeDistiilation(nn.Module, ABC):
    def __init__(self, cumbersome:str, cumbersome_config:dict, small:str, small_config:dict) -> None:
        super().__init__()

        model = importlib.import_module('models').__getattribute__(cumbersome)
        self.cumbersome_model:CumbersomeModel =  model(**cumbersome_config)

        model = importlib.import_module('models').__getattribute__(small)
        self.small_model:SmallModel =  model(**small_config)

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        cumbersome_outptus = self.cumbersome_model(*args, **kwargs)
        small_outptus = self.small_model(*args, **kwargs)
        return cumbersome_outptus, small_outptus
    
    @abstractmethod
    def loss_function(self, cumbersome_outputs:Tensor, small_outputs:Tensor, targets:Tensor)->Tensor:
        return None
    

class KnowledgeDistillation(AbsKnowledgeDistiilation):
    def loss_function(self, cumbersome_outputs: Tensor, small_outputs: Tensor, targets: Tensor) -> Tensor:
        return None
    

class TemporalKnowedgeDistillation(AbsKnowledgeDistiilation):
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return super().forward(*args, **kwargs)
    
    def loss_function(self, cumbersome_outputs: Tensor, small_outputs: Tensor, targets: Tensor) -> Tensor:
        return super().loss_function(cumbersome_outputs, small_outputs, targets)