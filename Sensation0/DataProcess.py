import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from debug_tools import Debug
from .AutoEncoder import Encoder
from .DeltaTime import DeltaT
from .configure import config
import torch
import numpy as np
from os.path import isfile
from torchvision import transforms

class DataEncoding:
    """
    This code is a part of Vision Sensation Process (Sensaion0).
    
    how to use.
    Ex.)
        >>> from Sensation0.DataProcess import DataEncoding
        >>> DE = DataEncoding(log_titile='test process',device='cpu')
        >>> img = read_image() # -> numpy.ndarray
        >>> encoded = DE.encode(img) # -> torch.Tensor
            ~ other process ~
        >>> delta_time = DE.calculate_distance(encoded, ReadOutMemory) # -> Tensor
    """
    def __init__(
        self,
        log_title:str,
        device:torch.device,
        encoder_params:str=config.encoder_params,
        deltatime_params:str=config.deltatime_params,
        dtype:torch.dtype=torch.float16,
        debug_mode:bool=False
        ) -> None:
        
        self.device = device
        self.debug_mode = debug_mode
        self.dtype = dtype
        self.debug = Debug(log_title,debug_mode)

        if not isfile(encoder_params):
            self.debug.exception(f'not exists {encoder_params}!')
        if not isfile(deltatime_params):
            self.debug.exception(f'not exists {deltatime_params}!')
        
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(encoder_params,map_location=device))
        self.encoder = self.encoder.type(dtype).to(device)
        
        self.deltaT = DeltaT()
        self.deltaT.load_state_dict(torch.load(deltatime_params,map_location=device))
        self.deltaT = self.deltaT.type(dtype).to(device)
        
        self.resizer = transforms.Resize(config.frame_size)
    @torch.no_grad()
    def encode(self,img:np.ndarray) -> torch.Tensor:
        """
        img: shape (height,width,channels)
        """
        img = torch.from_numpy(img).unsqueeze(0).to(self.device).permute(0,3,2,1)
        img = self.resizer(img).type(self.dtype)/255
        encoded = self.encoder(img).view(-1)
        return encoded

    @torch.no_grad()
    def calc_distance(self,encoded:torch.Tensor,ReadOutMemory:torch.Tensor) -> torch.Tensor:
        """
        encoded:shape (element,)
        ReadoutMemory:shape (length,element)
        """
        encdata = encoded.unsqueeze(0).to(self.device).repeat(ReadOutMemory.size(0),1)
        delta = self.deltaT(encdata,ReadOutMemory).view(-1)
        return delta

