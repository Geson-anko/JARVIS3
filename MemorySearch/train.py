import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from os.path import isfile

from MemoryManager import MemoryManager
import multiprocessing as mp
from .memory_search import MemorySearch
from .config import config
class Train(MemoryManager):
    LogTitle:str = f'Backup{MemorySearch.LogTitle}'

    def __init__(self,device:str=None,debug_mode:bool=False) -> None:
        super().__init__(log_title=self.LogTitle, debug_mode=debug_mode)

    def activation(self,shutdown:mp.Value,sleep:mp.Value) -> None:
        
        if isfile(config.dict_file):
            memdict = self.load_python_obj(config.dict_file)
            self.save_python_obj(config.backup_dict_file,memdict)
            self.log('Get backup.')
        else:
            self.log('MemoryDictionary does not exist.')

        del memdict