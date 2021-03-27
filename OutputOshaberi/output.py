import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import torch
import numpy as np

from OutputBase import OutputBase
from .config import config

class Output(OutputBase):
    LogTitle:str = 'OutputOshaberi'
    UsingMemoryFormat:str = '3'
    MaxMemoryLength:int = 1

    SleepWaitTime:float = 0.1
    MaxFrameRate:int = 30

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
    