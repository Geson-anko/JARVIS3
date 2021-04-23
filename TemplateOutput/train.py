import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from os.path import isfile
import numpy as np
import torch
from .config import config

from TrainBase import TrainBase

class Train(TrainBase):
    LogTitle:str = ''
    dtype:np.dtype = np.float16
    torchdtype:torch.dtype = torch.float16
    def TrainProcess(self) -> None:
        # ------ Additional Trainings ------
        #
        self.release_system_memory()
        # --- end of Additional Training ---

        self.log('Train process was finished')