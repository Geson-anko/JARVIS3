from dataclasses import dataclass
import os
import numpy as np
import torch
from datetime import timezone,timedelta

@dataclass
class Config:
    version:float= 3.0
    current_directory:str = os.path.dirname(os.path.abspath(__file__))
    log_dir:str = os.path.join(current_directory,'log') 
    logo_file:str = os.path.join(current_directory,'logo.txt')

    init_id:int = -1
    memory_folder:str = os.path.join(current_directory,'memory')
    active_time:float = 16.0
    ID_length:int = 10
    ID_dtype:str = 'int64'

    IDchars:str = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    decimal_base:int = len(IDchars)
    memory_file_form:str = os.path.join(memory_folder,'{0}.h5')

    ### switch
    wait_time:float = 0.1
    ### wait when sleep
    sleep_wait:float = 5
    ### timezone
    TimeZone:timezone = timezone(timedelta(hours=+9),name='JST')
    ### delta T threshold
    deltaT_threshold:float = 0.00001
    deltaT_zero_per:float = 0.1

    release_system_cache_time:float = 16 # second

    ### sensations
    
    """
    main process calls 'modulename.Sensation()'. so you must write like this below.
    modulename/__init__.py
    >>> from .sensation import Sensation
    """
    sensation_modules:tuple = (
        ('Sensation1','cuda'),
    )

    ### Trainer
    """
        Train process calls 'modulename.Train()'. so you must write like this below.
        modulename/__init__.py
        >>> from .train import Train    
    """
    train_modules:tuple = (
        "Sensation0",
    )
    train_wait:float = 5 #second
    



    ### datatypes
    int_types = set([int,np.int,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,torch.int,torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8])
    float_types = set([float,np.float16,np.float32,np.float64,torch.float16,torch.float32,torch.float64,torch.bfloat16])
    bool_types = set([bool,np.bool,torch.bool])
    str_types = set([str,np.str,np.str_])

