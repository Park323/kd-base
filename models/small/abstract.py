from abc import *
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

class SmallModel(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, inputs, input_lengths=None) -> Tensor:
        pass

    @abstractmethod
    def predict(self, inputs, input_lengths=None) -> Union[int, Tensor]:
        pass