# Knowledge Distillation

## Introduction

Implementation of training various models with knowledge distillation.


## Getting Started
### 0. Data prepare
 - Run jupyter notebooks located in **dataprepare**

### 1. train
```

```


### 2. test
```
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
