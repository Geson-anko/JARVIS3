import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import multiprocessing as mp
from typing import Tuple

from MemoryManager import MemoryManager
from MasterConfig import Config as mconf

from .config import config
from .GUI import GUIController

class Controller(MemoryManager):
    log_title:str = 'Controler'

    def __init__(self,debug_mode: bool=False) -> None:
        super().__init__(log_title=self.log_title, debug_mode=debug_mode)
    
    def activation(
        self,shutdown:mp.Value,switches:Tuple[Tuple[str,mp.Value],...],
        ) -> None:
        """
        shutdown_switch :    multiprocessing shared memory bool value.
        switch_objects  :    ((title, switch),...)
        """

        self.log('process started')
        while not shutdown.value:
            GUIController(shutdown,switches)()
        self.log('process shutdowned')

        

