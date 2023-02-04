import os
import sys
import tqdm

import torch

from dataloader.utils import load_dataset
from dataloader.dataset import DATA_LIST
from models.feats_extractor import *


def main(data_name:str, root:str='data', new_root:str='data2', subset:str=None, tag:str='_feat', whole_layers:bool=False):
    assert data_name in DATA_LIST, f"{data_name} IS NOT EXISTING DATASET!!"

    DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset(root=root, data_name=data_name, subset=subset) # should return (an audio array, sampling rate)

    extractor = Wav2VecExtractor().to(device=DEVICE)
    extractor.eval()
    for i in tqdm.tqdm(range(len(dataset))):
        new_path = dataset.generate_feature_path(i, new_root=new_root, tag=tag)
        waveform, sr = dataset[i]
        waveform = waveform.to(DEVICE)
        with torch.no_grad():
            features = extractor.extract(waveform, sr)
            features = [feature.detach().cpu() for feature in features]
        features = torch.cat(features, axis=0)
        if whole_layers:
            torch.save(features, new_path) # Save all hidden features (initial + 12 layers)
        else:
            torch.save(features[-1], new_path) # Save last layer features
        

if __name__=='__main__':
    main(*sys.argv[1:])