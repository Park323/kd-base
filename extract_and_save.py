import os
import sys
import tqdm

import torch

from dataloader.utils import load_dataset
from dataloader.dataset import DATA_LIST
from models import *


def main(args):
    DEVICE = 'cuda'if torch.cuda.is_available() else 'cpu'
    #===== Load Data =====
    dataset = load_dataset(root=root, data_name=data_name, url=url, subset=subset, ext='wav') # should return (an audio array, sampling rate)
    #===== Load Model ====
    extractor = Wav2VecExtractor().to(device=DEVICE)
    
    extractor.eval()
    for i in tqdm.tqdm(range(len(dataset))):
        new_path = dataset.generate_feature_path(i, new_root=new_root, tag=tag)
        waveform, sr = dataset[i]
        waveform = waveform.to(DEVICE)
        #===== Forward =====
        # cumbersome models have different extract methods, we can call it by 'getattr'.
        with torch.no_grad():
            outputs = getattr(extractor, 'option_instance_method_name')(waveform, sr)
            features = [feature.detach().cpu() for feature in features]
        features = torch.cat(features, axis=0)
        
        #===== Save outputs =====
        save_features = list()
        for layer_id in layer_ids:
            save_features.append(features[layer_id])
        save_features = torch.stack(save_features)
        torch.save(save_features, new_path)
        

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="Configuration file path")
    parser.add_argument('--model_path', type=str, help="Pre-trained model path")
    parser.add_argument('--save_path', type=str, help="Directory for saving extracted outputs")
    
    args = parser.parse_args()
    
    main(args)