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
    tempmem_scale_factor:float = 2
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
    """### delta T threshold # delta Time model was ...
    deltaT_threshold:float = 0.00001
    deltaT_zero_per:float = 0.1
    """

    release_system_cache_time:float = 16 # second

    ### processer settings
    max_processes:int = 4
    main_proceess_group_number:int = 0
    """
    max_processes is to save CPU,GPU, and each Memories.
    please set process groups. default main process group number is 0
    Ex.
    >>> processes = [[],] * os.cpu_count
    >>> processes[YourProcessGroupNumber].append((func,args_tuple))
    """
    ### sensations
    
    """
    main process calls 'modulename.Sensation()'. so you must write like this below.
    modulename/__init__.py
    >>> from .sensation import Sensation
    """
    sensation_modules:tuple = ( #(Module_name, device, process group number)
        ('Sensation1','cuda',1),
        ('Sensation2','cuda',0),
        #('Sensation3','cuda',1),
        ('Sensation4','cuda',1),
        ('Sensation5','cuda',0),
        ('Sensation6','cuda',1)
    )

    ### outputs
    all_true_wait_time:float = 0.01
    output_modules:tuple = ( # (Module_name, device, process group number)
        #('OutputVision','cuda',0),
        ('OutputMouse','cpu',2),
        ('OutputOshaberi','cuda',0)
    )

    ### Trainer
    """
        Train process calls 'modulename.Train()'. so you must write like this below.
        modulename/__init__.py
        >>> from .train import Train    
    """
    device_trainer = 'cuda'
    train_modules:tuple = (
        "Sensation1",
        "Sensation2",
        "Sensation6",
        "MemorySearch",
        "outputOshaberi",
    )
    train_wait:float = 5 #second
    



    ### datatypes
    int_types = set([int,np.int,np.int8,np.int16,np.int32,np.int64,np.uint8,np.uint16,np.uint32,np.uint64,torch.int,torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8])
    float_types = set([float,np.float16,np.float32,np.float64,torch.float16,torch.float32,torch.float64,torch.bfloat16])
    bool_types = set([bool,np.bool,torch.bool])
    str_types = set([str,np.str,np.str_])

