import torch
from torch.nn import (
    Module,Linear,
    Conv1d,BatchNorm1d,AvgPool1d,
    ConvTranspose1d,Upsample,
)
import os
import random

from .config import config

class Encoder(Module):
    input_size:tuple = (1,config.futures,config.length,2)
    output_size:tuple = (1,64,8)
    insize = (-1,*input_size[1:])
    def __init__(self):
        super().__init__()
        self.reset_seed()

        # Model layers
        self.pool = AvgPool1d(2)
        self.dense = Linear(2,1)
        self.conv1 = Conv1d(513,256,4)
        self.norm1 = BatchNorm1d(256)
        self.conv2 = Conv1d(256,128,8)
        self.norm2 = BatchNorm1d(128)
        self.conv3 = Conv1d(128,64,4)
        self.norm3 = BatchNorm1d(64)

    def forward(self,x):
        x = x.view(self.insize)
        x = torch.relu(self.dense(x)).squeeze(-1)
        x = torch.relu(self.norm1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.norm2(self.conv2(x)))
        x = self.pool(x)
        x = torch.tanh(self.norm3(self.conv3(x)))
        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
class Decoder(Module):
    input_size = Encoder.output_size
    output_size = Encoder.input_size
    insize = (-1,*Encoder.output_size[1:])
    def __init__(self):
        super().__init__()
        self.reset_seed()

        # Model layers
        self.upper = Upsample(scale_factor=2)
        self.dcon0 = ConvTranspose1d(64,128,7)
        self.norm0 = BatchNorm1d(128)
        self.dcon1 = ConvTranspose1d(128,256,3)
        self.norm1 = BatchNorm1d(256)
        self.dcon2 = ConvTranspose1d(256,513,4)
        self.norm2 = BatchNorm1d(513)
        self.dense = Linear(1,2)

    def forward(self,x):
        x = x.view(self.insize)
        x = torch.relu(self.norm0(self.dcon0(x)))
        x = self.upper(x)
        x = torch.relu(self.norm1(self.dcon1(x)))
        x = self.upper(x)
        x = torch.relu(self.norm2(self.dcon2(x))).unsqueeze(-1)
        x = self.dense(x)
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


"""