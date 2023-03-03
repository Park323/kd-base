import os
from typing import List, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import SpeechCommandDataset, ESC50


def load_dataset(data_name:str='speechcommands', get_collate_fn:bool=False, **kwargs)->Tuple[torch.utils.data.Dataset, Optional[Callable]]:    
    """
    get_collate_fn=False option is used at *_extract_feats.py
    for extracting a single data
    """
    if data_name == 'speechcommands':
        for key in ['root', 'subset']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = SpeechCommandDataset(**kwargs)
        collate_fn = wav_pad_collate
    elif data_name == 'esc50':
        for key in ['fold']:
            assert key in kwargs, f"Pass '{key}' through the config yaml file!!"
        dataset = ESC50(**kwargs)
        collate_fn = wav_pad_collate
    else:
        assert False, f"DATA '{data_name}' IS NOT IMPLEMENTD!"

    if get_collate_fn:
        return dataset, collate_fn
    else: 
        return dataset

def wav_pad_collate(batch:List[Tuple[Tensor, int]]):
    # batch[0]: [B, C, T], C==1 (single channel audio)
    # e.g. [B, 1, T]
    B = len(batch)
    batch_sample = batch[0][0] # [C, T]
    batch_dim = len(batch_sample.shape) # 2 (@ using wav file)

    assert batch_dim == 2, "Audio data should be 2D tensor (C, T) where C==1 for single channel."

    search_dim = 1 if batch_dim == 2 else 0 # search_dim = 1

    C, T = batch_sample.shape

    data_lengths = torch.zeros((B,), dtype=torch.long)
    for i, (array, _) in enumerate(batch): # array: [C, T]
        data_lengths[i] = array.size(search_dim) # search_dim = 1 (T)
    max_array_length = data_lengths.max()

    data = torch.zeros((B, max_array_length)) # [B, T_MAX]
    labels = torch.zeros((B, ), dtype=torch.long)

    for i, (array, label) in enumerate(batch):

        data[i, :array.size(1)] = array.squeeze(dim = 0)
        labels[i] = label

    return data, data_lengths, labels

