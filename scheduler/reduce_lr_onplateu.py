#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

class ReduceLROnPlateau(ReduceLROnPlateau):
	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)
		self.interval = 'epoch'
		self.frequency = 1

def Scheduler(optimizer, factor, patience, min_lr, threshold, **kwargs):

	sche_fn = ReduceLROnPlateau(optimizer,
							 'min', factor = factor, 
							 patience=patience, 
							 min_lr=min_lr, 
							 threshold=threshold, 
							 verbose=1)

	print('Initialised ReduceLROnPlateau scheduler')

	return sche_fn