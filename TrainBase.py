import numpy as np
import torch

from torch_model_fit import Fit
from MemoryManager import MemoryManager
import multiprocessing as mp
from typing import Tuple
from MasterConfig import Config


class TrainBase(MemoryManager):
    LogTitle:str

    def __init__(self,device:torch.device,debug_mode:bool=False) -> None:
        super().__init__(log_title=self.LogTitle, debug_mode=debug_mode)
        self.device = torch.device(device)
        self.fit = Fit(self.LogTitle,debug_mode)

    def activation(self,shutdown:mp.Value,sleep:mp.Value) -> None:
        self.shutdown = shutdown
        self.sleep = sleep

        self.TrainProcess()

        self.release_system_memory()
        self.log('finished all trainings.')

    def TrainProcess(self) -> None:pass

    def Train(self,*args,**kwargs) -> Tuple[torch.Tensor,...]:
        return self.fit.Train(self.shutdown,self.sleep,*args,**kwargs)

