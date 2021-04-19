import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import math
import torch
from torch.nn import Module
from os.path import join as pathjoin

from .config import config
from SensationBase import SensationBase
from .sensation_models import Encoder

class Sensation(SensationBase):
    MemoryFormat:str = '_'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = 16384 # ReadOutLength
    KeepLength:int = math.floor(ReadOutLength*0.7)  # ReadOutLength * 0.7
    MemoryListLength:int = math.floor(ReadOutLength*0.01)# 1% of ReadOutLength
    MemorySize:int = int(np.prod(Encoder.output_size))
    SameThreshold:float = 0.001 # The Threshold of memory error.
    DataSize:tuple = Encoder.input_size[1:]
    DataSaving:bool = True
    DataSavingRate:int = 64

    Encoder:Module = Encoder
    SleepWaitTime:float = 0.1

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    Data_folder:str = pathjoin(Current_directory,'data') # /current_dicrectory/data/
    Temp_folder:str = pathjoin(Current_directory,'temp') # /Current_directory/temp/

    ## defining parameter file
    Encoder_params:str= pathjoin(Param_folder,'encoder.params') # your encoder parameter file name
    Decoder_params:str = pathjoin(Param_folder,'decoder.params') # yout decoder parameter file name

    ## defining temporary file
    NewestId_file:str = pathjoin(Temp_folder,'NewestId.pkl')
    MemoryList_file:str = pathjoin(Temp_folder,'MemoryList.pkl')
    ReadOutMemory_file:str = pathjoin(Temp_folder,'ReadOutMemory.pkl')
    ReadOutId_file:str = pathjoin(Temp_folder,'ReadOutId.pkl')
    ReadOutTime_file:str = pathjoin(Temp_folder,'ReadOutTime.pkl')

    dtype:np.dtype = np.float16  # data type
    torchdtype:torch.dtype = torch.float16 # torch tensor data type

    # ------ your settings ------
    def LoadModels(self) -> None:
        super().LoadModels()
        pass # This method is called when sleep Time and regularly.

    def Start(self) -> None:
        # This method is called when process start.
        pass

    def Update(self) -> torch.Tensor:
        # This method is called every frame start.
        # your data process
        return 'your data. type is torch.Tensor'

    def UpdateEnd(self) -> None:
        # This method is called every frame end.
        pass

    def End(self) -> None:
        # This method is called when shutting down process.
        pass

    # ------ train settings ------
    Training_dtype:torch.dtype = torch.float16
    AutoEncoderDataSize:int = 8192
    AutoEncoderLearningRate:float = 0.0001
    AutoEncoderBatchSize:int = 2**0
    AutoEncoderEpochs:int = 4


