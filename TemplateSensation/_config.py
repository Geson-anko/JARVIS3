from dataclasses import dataclass
import os
from os.path import join as pathjoin
import numpy as np
import torch

@dataclass
class config:
    current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /current_directory/...  from root
    param_folder:str = pathjoin(current_directory,'params') # /current_directory/params/
    data_folder:str = pathjoin(current_directory,'data') # /current_dicrectory/data/
    temp_folder:str = pathjoin(current_directory,'temp') # /current_directory/temp/

    ## defining parameter file
    encoder_params:str= pathjoin(param_folder,'_') # your encoder parameter file name
    decoder_params:str = pathjoin(param_folder,'_') # yout decoder parameter file name
    deltatime_params:str = pathjoin(param_folder,'_') # your deltatime parameter file name

    ## defining temporary file
    NewestId_file:str = pathjoin(temp_folder,'NewestId.pkl')
    MemoryList_file:str = pathjoin(temp_folder,'MemoryList.pkl')
    ReadOutMemory_file:str = pathjoin(temp_folder,'ReadOutMemory.pkl')
    ReadOutId_file:str = pathjoin(temp_folder,'ReadOutId.pkl')
    ReadOutTime_file:str = pathjoin(temp_folder,'ReadOutTime.pkl')

    ## defining data file
    """default None."""

    ReadOutLength:int = -1 # Your ReadOutId and Memory Length. Length > 0
    KeepLength:int = -1 # 0 < KeepLength < ReadOutLength.
    MemoryListLength:int = int(np.floor(ReadOutLength * 0.01)) # 1% of ReadOutLength is default. 
    wait_time:float = 0.1 # 1/FPS your sensation

    dtype:np.dtype = np.float16 # data type
    torchdtype:torch.dtype = torch.float16 # torch tensor data type


""" Documentation
This code is a config for your sensation process and others. Please write here what you need for your programs.
More detailed examples are written in Sensation0/configure.py

** default constants **

current_directory:
    The path of your sensation folder. If you want to use files for other programs,
    please use the following.
    Ex.)
    >>> folder = os.path.join(current_directory,'YourFolder')
    >>> file = os.path.join(current_directory,'YourFile')

param_folder:
    The folder path for AutoEncoder and DeltaTime parameter files that were pretrained.
    Please use the following.
    Ex.)
    >>> encoder_file:str = os.path.join(param_folder,'encoder.params')

data_folder:
    This is a folder for training data.

temp_folder:
    This folder is for temporary file. please use it to save data when process is shut down.

ReadOutLength:
    This is ReadOutMemory and Id Length. This is calucalated from wait_time.
    You must experiment to see how it increases.

KeepLength:




"""