import os
import re
import glob
from pathlib import Path
from typing import Tuple, Union, Optional

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor
import pandas as pd
import librosa
from torch.utils.data import Dataset


HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
DATA_LIST = [
    "speechcommands",
]

# Keyword Spotting
class SpeechCommandDataset(torchaudio.datasets.SPEECHCOMMANDS):
    CLASS_LIST = [
        'backward', 'bed', 'bird', 'cat', 'dog', 'down', 
        'eight', 'five', 'follow', 'forward', 'four', 'go', 
        'happy', 'house', 'learn', 'left', 'marvin', 'nine', 
        'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 
        'six', 'stop', 'three', 'tree', 'two', 'up', 
        'visual', 'wow', 'yes', 'zero']
    CLASS_DICT = {class_:i for i, class_ in enumerate(CLASS_LIST)}
    CLASS_DICT_INV = {value:key for key, value in CLASS_DICT.items()}

    def __init__(self, root:str='data', folder_in_archive='SpeechCommands', url='speech_commands_v0.02', subset:str='training', ext:str='wav', download=False):
        super().__init__(subset=subset, root=root, folder_in_archive=folder_in_archive, url=url, download=download)
        assert subset in ['training','validation','testing']
        assert os.path.exists(root+'/'+folder_in_archive)

        self.folder_in_archive = folder_in_archive
        self.ext = ext

        if subset == "validation": pass
        elif subset == "testing": pass
        elif subset == "training":
            excludes = set(_load_list(self._path, "validation_list.txt", "testing_list.txt"))
            walker = sorted(str(p) for p in Path(self._path).glob(f"*/*.{self.ext}"))
            self._walker = [
                w
                for w in walker
                if HASH_DIVIDER in w and EXCEPT_FOLDER not in w and os.path.normpath(w) not in excludes
            ]
        else:
            walker = sorted(str(p) for p in Path(self._path).glob(f"*/*.{self.ext}"))
            self._walker = [w for w in walker if HASH_DIVIDER in w and EXCEPT_FOLDER not in w]
        
    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        return super().__getitem__(n)[:2] #Tuple[Tensor, int, str, str, int]

    def get_metadata(self, index):
        return self._walker[index]

    def label2index(self, label):
        return self.CLASS_DICT[label]

    def index2label(self, index):
        return self.CLASS_DICT_INV[index]

    def generate_feature_path(self, index, new_root:str=None, tag:str='_feat'):
        old_path = self.get_metadata(index)
        new_path = old_path.replace(self.folder_in_archive, self.folder_in_archive + tag).replace('.wav','.pt')
        
        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        return new_path



class ESC_Dataset(Dataset):
    def __init__(self, audio_path, meta_data:str='./esc50.csv', fold:list=[1, 2, 3, 4, 5]):
        self.meta_data = pd.read_csv(meta_data) # esc50.csv
        self.fold = fold # [1, 2, 3, 4, 5]
        self.audio_path = audio_path # .../esc-50/resample

        # set dataset using the fold
        self.meta_data = self.meta_data.loc[self.meta_data['fold'].isin(self.fold)] 

        
        print(f"fold: {self.fold}")
        print(f"dataset length: {len(self.meta_data)}")
        
        # reset index
        self.meta_data.reset_index(drop = True, inplace=True)

    def __getitem__(self, index):
        name = self.meta_data.loc[index, 'filename']
        audio, sr = torchaudio.load(os.path.join(self.audio_path, name)) # 
        y = self.meta_data.loc[index, 'target']

        return audio, y

    def __len__(self):
        return len(self.meta_data)
