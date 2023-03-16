import logging
from typing import Any, Mapping
import pytorch_lightning as pl
import torch


def load_engine(engine_type:str, *args, **kwargs)->pl.LightningModule:
    if engine_type:
        Engine_cls = TrainEngine
    elif True:
        Engine_cls = TrainEngine
    else:
        raise NotImplementedError
    return Engine_cls(*args, **kwargs)

def acc_step(y_hat, y):
    if isinstance(y_hat, tuple): y_hat = y_hat[0] # Get small model's output
    correct = (y_hat.argmax(1) == y).type(torch.float).sum().item()
    size = y.shape[0]
    return {'correct': correct, 'size': size}

def acc_calculate(step_outputs:dict, name:str='val'):
    correct_score = sum([dic['correct'] for dic in step_outputs])
    total_size = sum([dic['size'] for dic in step_outputs])
    acc = correct_score/total_size
    return name+'_ACC', acc*100

def test_loss_switch(func, metric:str):
    # Switch loss function, only for evaluation
    return func

MetricFuncs = dict(
    acc=dict(
        step_func=acc_step,
        epoch_func=acc_calculate
    ),
)


class TrainEngine(pl.LightningModule):
    def __init__(self, model, loss_function, optimizer, scheduler, metric='acc', **kwargs):
        super().__init__()
        # ⚡ model
        self.model = model
        print(self.model)

        # ⚡ loss 
        self.loss_function = model.loss_function if hasattr(model, 'loss_function') else loss_function

        # ⚡ optimizer
        self.optimizer = optimizer

        # ⚡ scheduler
        self.scheduler = scheduler # **kwargs: **config['scheduler_config']

        # save hyperparameters
        self.save_hyperparameters(ignore=['model', 'loss_function', 'optimizer', 'scheduler'])

        #⚡⚡⚡ debugging - print input output layer ⚡⚡⚡
        sample_size = tuple(map(int, kwargs['sample_input_size'].split())) if kwargs.get('sample_input_size', False) else (64,1,28,28)
        self.example_input_array = torch.randn(sample_size)

        # custom
        self.metric = metric


    def training_step(self, batch, batch_idx): # batch : (tensor[B, T], data_length, label)

        x = batch[:-1] # (tensor[B, T], data_length)
        y = batch[-1] # label
        # preprocess
        
        # inference
        y_hat = self.model(*x)

        # post processing

        # calculate loss
        loss = self.loss_function(y_hat, y)
        # Logging to TensorBoard
        self.log("loss", loss, on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[:-1]
        y = batch[-1]

        y_hat = self.model(*x)
        loss_function = test_loss_switch(self.loss_function, self.metric)
        loss = loss_function(y_hat, y)
        self.log("test_loss", loss,  on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        return MetricFuncs[self.metric]['step_func'](y_hat, y)

    def test_epoch_end(self, test_step_outputs):
        name, value = MetricFuncs[self.metric]['epoch_func'](test_step_outputs, name='test')
        self.log(name, value, on_epoch = True, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[:-1]
        y = batch[-1]

        y_hat = self.model(*x)
        loss_function = test_loss_switch(self.loss_function, self.metric)
        loss = loss_function(y_hat, y)
        self.log("val_loss", loss,  on_epoch= True, prog_bar=True, logger=True, sync_dist=True)

        return MetricFuncs[self.metric]['step_func'](y_hat, y)

    def validation_epoch_end(self, validation_step_outputs):
        name, value = MetricFuncs[self.metric]['epoch_func'](validation_step_outputs, name='val')
        self.log(name, value, on_epoch = True, prog_bar=True, sync_dist=True)
        
        lr =  self.optimizers().param_groups[0]['lr']
        self.log("lr", lr, on_epoch=True, logger=True, prog_bar=True)

    def forward(self, x):
        y_hat = x
        return y_hat

    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "val_loss",
                "interval": self.scheduler.interval,
                "frequency": self.scheduler.frequency,
                "name": 'lr_log'
            },
        }