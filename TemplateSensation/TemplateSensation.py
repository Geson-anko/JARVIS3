from ..MemoryManager import MemoryManager
from ..Config import Config
from ..AutoEncoders.AutoEncoder_ import DataEncoding, DeltaTime

import torch
import *** # using libs

class Sensation_(MemoryManager):
    memory_format:str = '_'
    log_title:str = f'sensation{memory_format} : '

    def __init__(self,device:str or torch.device):
        super().__init__(self.log_title)
        self.device = torch.device(device)

    def activation(self,cmd,sleep,ReadOutId,memory_list,newest_id,*optionals=None):
        encoding = DataEncoding(device=self.device)
        