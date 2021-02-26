import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from MemoryManager import MemoryManager
from MasterConfig import Config as mconf
from .config import config

class SleepManager(MemoryManager):
    def __init__(self, log_title: str, debug_mode: bool=False) -> None:
        super().__init__(log_title=log_title, debug_mode=debug_mode)
        
