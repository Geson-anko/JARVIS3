from dataclasses import dataclass
import os
from os.path import join as pathjoin
import math
import torch
import numpy as np

@dataclass
class config:
    frame_rate:int = 30
    width:int = 640
    height:int = 360
    channels:int = 3
    frame_size:tuple = (width,height)

    dtype:np.dtype = np.float16
    torchdtype:torch.dtype = torch.float16

    current_directory:str = os.path.dirname(os.path.abspath(__file__))

    param_folder:str = pathjoin(current_directory,'params')
    encoder_params:str = pathjoin(param_folder,'encoder.params')
    deltatime_params:str = pathjoin(param_folder,'deltatime.params')
    decoder_params:str = pathjoin(param_folder,'decoder.params')
    
    temp_folder:str = pathjoin(current_directory,'temp')
    newestId_file:str = pathjoin(temp_folder,'newestId.pkl')
    memlist_file:str = pathjoin(temp_folder,'memory_list.pkl')
    ReadoutMem_file:str = pathjoin(temp_folder,'ReadoutMem.pkl')
    ReadoutId_file:str = pathjoin(temp_folder,'ReadoutId.pkl')
    ReadoutTime_file:str = pathjoin(temp_folder,'ReadoutTime.pkl')

    data_folder:str = pathjoin(current_directory,'data')
    video_data:str = pathjoin(data_folder,'video.h5')

    default_video_capture:int = 0

    ReadoutLength:int = 10000
    KeepLength:int = int(math.floor(ReadoutLength * 0.7))
    MemListLength:int = 100
    wait_time:float = 0.3
    saving_rate:int = 128

    # train.py
    training_dtype:torch.dtype = torch.float16
        # AutoEncoder setting
    train_video_use:int = int(np.floor(8192/saving_rate))
    AE_lr:float = 0.0001
    AE_batch_size:int = 32
    AE_epochs:int = 5
        # DeltaTime settings
    time_use:int = 8192
    DT_lr:float = 0.001
    DT_batch_size:int = 4096
    DT_epochs:int = 10
    zero_per:float = 0.1