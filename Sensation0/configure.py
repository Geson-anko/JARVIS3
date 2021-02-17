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

    dtype:np.dtype = np.float32
    torchdtype:torch.dtype = torch.float32

    current_directory:str = os.path.dirname(os.path.abspath(__file__))

    encoder_params:str = pathjoin(current_directory,'params/encoder.params')
    deltatime_params:str = pathjoin(current_directory,'params/deltatime.params')
    decoder_params:str = pathjoin(current_directory,'params/decoder.params')
    
    newestId_file:str = pathjoin(current_directory,'temp/newestId.pkl')
    memlist_file:str = pathjoin(current_directory,'temp/memory_list.pkl')
    ReadoutMem_file:str = pathjoin(current_directory,'temp/ReadoutMem.pkl')
    ReadoutId_file:str = pathjoin(current_directory,'temp/ReadoutId.pkl')
    ReadoutTime_file:str = pathjoin(current_directory,'temp/ReadoutTime.pkl')

    video_data:str = pathjoin(current_directory,'data/video.h5')

    default_video_capture:int = 0

    ReadoutLength:int = 10000
    KeepLength:int = int(math.floor(ReadoutLength * 0.7))
    MemListLength:int = 100
    wait_time:float = 0.3