import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import math
import torch
from torch.nn import Module,BatchNorm2d 
from os.path import join as pathjoin

from .config import config
from SensationBase import SensationBase
from .sensation_models import Encoder

import pyaudio

class Sensation(SensationBase):
    MemoryFormat:str = '2'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = 16384 # ReadOutLength
    KeepLength:int = math.floor(ReadOutLength*0.7)  # ReadOutLength * 0.7
    MemoryListLength:int = math.floor(ReadOutLength*0.01)# 1% of ReadOutLength
    MemorySize:int = int(np.prod(Encoder.output_size))
    SameThreshold:float = 0.13 # The Threshold of memory error.
    DataSize:tuple = Encoder.input_size[1:]
    DataSavingRate:int = 64

    Encoder:Module = Encoder
    SleepWaitTime:float = 0.25

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

    mins = 0.0

    # ------ your settings ------
    

    def Start(self) -> None:
        # This method is called when process start.
        self.audio =pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=config.pyaudio_format,
            channels=config.channels,
            rate=config.frame_rate,
            frames_per_buffer=config.CHUNK,
            input=True,
        )
        self.log('Audio streamer is ready.')
        self.sound = np.zeros((config.sample_length,),dtype=config.audio_dtype)
        self.norm = BatchNorm2d(config.futures,track_running_stats=False).to(self.device)
        


    def Update(self) -> torch.Tensor:
        # This method is called every frame start.
        # your data process
        data = self.stream.read(config.CHUNK)
        data = np.frombuffer(data,config.audio_dtype).reshape(-1)
        self.sound[:-config.CHUNK] = self.sound[config.CHUNK:].copy()
        self.sound[-config.CHUNK:] = data
        data = self.sound / config.sample_range
        data = torch.from_numpy(data).type(torch.float32).to(self.device)
        data = torch.stft(data,config.n_fft,config.hop_length,return_complex=False).unsqueeze(0)
        data = self.norm(data)
        #print('\rcurrent_length',self.current_length,'mins',self.mins,end='')
        return data

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
    AutoEncoderBatchSize:int = 1024
    AutoEncoderEpochs:int = 4


