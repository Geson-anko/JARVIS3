from .config import config
import torch
from torch.nn import (
    Module, 
)
import os
import random

class DeltaT(Module):
    def __init__(self):
        super().__init__()
        self.reset_seed()

        # Model layers

    def forward(self,x):

        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

""" Documentation
This code is a Delta Time Model for your DataProcess.
The DeltaTime Model exists to calculate the distance between memories. 
More detailed examples are written in Sensation0/DeltaTime.py

"""