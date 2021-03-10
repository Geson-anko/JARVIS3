import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import torch
from torch.nn import Module
from os.path import join as pathjoin

from SensationBase import SensationBase
from .sensation_models import Encoder,DeltaTime

class Sensation(SensationBase):
    MemoryFormat:str = '_'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = -1 # ReadOutLength
    KeepLength:int = -1 # ReadOutLength * 0.7
    MemoryListLength:int = -1
    MemorySize:int = int(np.prod(Encoder.output_size))
    DataSize:int = int(np.prod(Encoder.input_size))
    DataSavingRate:int = 32

    Encoder:Module = Encoder
    DeltaTime:Module = DeltaTime
    SleepWaitTime:float = 0.1

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    data_folder:str = pathjoin(Current_directory,'data') # /current_dicrectory/data/
    Temp_folder:str = pathjoin(Current_directory,'temp') # /Current_directory/temp/

    ## defining parameter file
    Encoder_params:str= pathjoin(Param_folder,'_') # your encoder parameter file name
    Decoder_params:str = pathjoin(Param_folder,'_') # yout decoder parameter file name
    DeltaTime_params:str = pathjoin(Param_folder,'_') # your deltatime parameter file name

    ## defining temporary file
    NewestId_file:str = pathjoin(Temp_folder,'NewestId.pkl')
    MemoryList_file:str = pathjoin(Temp_folder,'MemoryList.pkl')
    ReadOutMemory_file:str = pathjoin(Temp_folder,'ReadOutMemory.pkl')
    ReadOutId_file:str = pathjoin(Temp_folder,'ReadOutId.pkl')
    ReadOutTime_file:str = pathjoin(Temp_folder,'ReadOutTime.pkl')

    dtype:np.dtype = np.float16  # data type
    torchdtype:torch.dtype = torch.float16 # torch tensor data type


    def Start(self) -> None:
        pass

    def Update(self) -> torch.Tensor:
        # your data process
        return 'your data. type is torch.Tensor'

    def UpdateEnd(self) -> None:
        pass

    def End(self) -> None:
        pass

    # ------ train settings ------
    Training_dtype:torch.dtype = torch.float16
    AutoEncoderDataSize:int = 8192
    AutoEncoderLearningRate:float = 0.0001
    AutoEncoderBatchSize:int = 2**0
    AutoEncoderEpochs:int = 1
    
    DeltaTimeDataSize:int = 8192
    DeltaTimeLearningRate:float = 0.0001
    DeltaTimeBatchSize:int = 2**0
    DeltaTimeEpochs:int = 1
    

