from dataclasses import dataclass
from dataclasses import dataclass
import os
from os.path import join as pathjoin
from dataclasses import dataclass

@dataclass
class config:
    current_directory:str = os.path.dirname(os.path.abspath(__file__))
    dict_file:str = pathjoin(current_directory,'MemoryDictionary.pkl')
    tempmem_file:str = pathjoin(current_directory,'tempmem.pkl')
    
    max_connection:int =7
    wait_time:float = 0.1



