import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from os.path import join as pathjoin
import torch
import numpy as np

from OutputBase import OutputBase
from .config import config

class Output(OutputBase):
    LogTitle:str = 'Output_'
    UsingMemoryFormat:str = '1'
    MaxMemoryLength:int = 1
    UseMemoryLowerLimit:int = 1

    SleepWaitTime:float = 0.1
    MaxFrameRate:int = 30

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root

    dtype = np.float16
    torchdtype = torch.float16

    def LoadModels(self) -> None:
        pass

    def Start(self) -> None:
        pass

    def Update(self, MemoryData: torch.Tensor) -> None:
        pass

    def UpdateEnd(self) -> None:
        pass

    def End(self) -> None:
        pass
    