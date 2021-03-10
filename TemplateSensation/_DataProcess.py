import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from debug_tools import Debug
from .AutoEncoder import Encoder
from .DeltaTime import DeltaT
from .config import config

import torch
import numpy as np
from typing import Any

class DataEncoding:
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

        if not os.path.isfile(encoder_params):
            self.debug.exception(f'not exists {encoder_params}!')
        if not os.path.isfile(deltatime_params):
            self.debug.exception(f'not exists {deltatime_params}!')
        
        self.encoder = Encoder()
        self.encoder.load_state_dict(torch.load(encoder_params,map_location=device))
        self.encoder = self.encoder.type(dtype).to(device)
        
        self.deltaT = DeltaT()
        self.deltaT.load_state_dict(torch.load(deltatime_params,map_location=device))
        self.deltaT = self.deltaT.type(dtype).to(device)

    @torch.no_grad()
    def encode(self,data:Any)-> torch.Tensor:
        """ 
        your process 
        """
        encoded = self.encoder(data).view(-1)
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


""" Documentaion
This code is part of Sensation Process. 
But this is not always necessary because it's relativery easy to implement in sesation.py
The DataEncoding class has 2 advantage. 
    First, you will never forget 'with torch.no_grad()'.
    Second, it helps you to derive ReadOutLength.



More detailed examples are written in Sensation0/DataProcess.py
"""