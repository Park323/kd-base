import torch
import torch.nn.functional as F

def loss_function(y_hat, target):
    return -torch.mm(y_hat, target)