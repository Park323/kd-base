# Knowledge Distillation

## Introduction

Implementation of training various models with knowledge distillation.


## Getting Started

### 0. feature extraction (Optional)
```
python pre_extract_feats.py iemocap /home/nas4/DB/IEMOCAP /home/nas4/DB/IEMOCAP/IEMOCAP None None _feat_1_12

# IC - train
python pre_extract_feats.py fluentspeechcommand /home/nas4/DB/fluent_speech_commands /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset None train _feat_1_12

# IC -valid
python pre_extract_feats.py fluentspeechcommand /home/nas4/DB/fluent_speech_commands /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset None valid _feat_1_12

# IC - test
python pre_extract_feats.py fluentspeechcommand /home/nas4/DB/fluent_speech_commands /home/nas4/DB/fluent_speech_commands/fluent_speech_commands_dataset None test _feat_1_12

```


### 1. train
```
# EMOTION RECOGNITION
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/er_avg.yaml --mode train
```


### 2. test
```
CUDA_VISIBLE_DEVICES=0 python main.py --config ./configs/esc_50.yaml --mode test
```


## Contribution Guide

1. All models should be placed at `./models`. They could be either separated py file or module directory, but they should be defined at `./models/__init__.py` so that other program can call them without indicating their subdirectories.

1. Two Dataloader mode exists and those can be switched by argument \
   Extracting outputs from cumbersome model (1) on the fly, (2) on the memory\ 
   So all cumbersome models should have method `extract` which can gets various arguments. \
   `extract_and_save.py` call `extract` method from cumbersome pretrained model and save them.

1. Model's initializing arguments should be defined at yaml config file.
   Same configuration file can be used at both `extract_and_save.py` and `main.py`

1. Distillation use one model which contains both cumbersome and small model. \
   Their parameters, pretrained path and all other options for initializing each of models are supplied by a yaml file. \
   Those are implemented in `./models/kd.py`