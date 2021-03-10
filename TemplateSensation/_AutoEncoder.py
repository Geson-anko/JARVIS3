from .config import config
import torch
from torch.nn import (
    Module, 
)
import os
import random

class Encoder(Module):
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
    
class Decoder(Module):
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

class AutoEncoder(Module):
    def __init__(self):
        """
        This class is used training only. 
        How about using it like this?
            >>> model = AutoEncoder()
            >>> # -- Training Model Process --
            >>> torch.save(model.encoder.state_dict(),encoder_name)
            >>> torch.save(model.decoder.state_dict(),decoder_name)
        """
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

""" Documentation

This code is a AutoEncoder for your DataProcess. 
The AutoEncoder Model exists to make the information smaller. So your DataProcess will use Encoder only.
More detailed examples are written in Sensation0/AutoEncoder.py


"""