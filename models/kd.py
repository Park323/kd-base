import importlib
from abc import *
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from .small import SmallModel
from .cumbersome import CumbersomeModel

from loss.contrastive import loss_function as contrastive_loss
from loss.softmax import loss_function as ce_loss


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
    def loss_function(self, outputs:Tuple[Tensor,Tensor], targets: Tensor) -> Tensor:
        return None
    

class KnowledgeDistillation(AbsKnowledgeDistiilation):
    def __init__(self, cumbersome: str, cumbersome_config: dict, small: str, small_config: dict, a:float=0.5) -> None:
        super().__init__(cumbersome, cumbersome_config, small, small_config)
        self.a = a

    def loss_function(self, outputs:Tuple[Tensor,Tensor], targets: Tensor) -> Tensor:
        cumbersome_outputs, small_outputs = outputs
        loss_cot = contrastive_loss(cumbersome_outputs, small_outputs)
        loss_ce = ce_loss(small_outputs, targets)
        return self.a * loss_cot + (1-self.a) * loss_ce
    

class TemporalKnowedgeDistillation(AbsKnowledgeDistiilation):
    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        return super().forward(*args, **kwargs)
    
    def loss_function(self, outputs: Tuple[Tensor, Tensor], targets: Tensor) -> Tensor:
        return super().loss_function(outputs, targets)