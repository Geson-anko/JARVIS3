from dataclasses import dataclass
import os
import numpy as np
import torch

@dataclass
class Config:
    version:float= 3.0
    log_dir:str = 'log'


    memory_folder:str = 'memory'
    active_time:float = 16.0
    ID_length:int = 10
    memory_search_max_roop:int = 2**20
    a_memlist_lim:int = 500
    brunch :int = 7

    IDchars:str = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    decimal_base:int = len(IDchars)
    memory_file_form:str = memory_folder+'/{0}.h5'

    ### command values
    shutdown:int = 999
    force_sleep:int = 123
    wake:int = 0
    ###



    current_directory:str = os.path.dirname(os.path.abspath(__file__))
    



    ### datatypes
    int_types = set([int,np.int,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,torch.int,torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8])
    float_types = set([float,np.float16,np.float32,np.float64,torch.float16,torch.float32,torch.float64,torch.bfloat16])
    bool_types = set([bool,np.bool,torch.bool])
    str_types = set([str,np.str,np.str_])