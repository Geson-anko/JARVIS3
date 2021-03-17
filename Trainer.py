import time
import multiprocessing as mp
import torch
import importlib

from MasterConfig import Config
from MemoryManager import MemoryManager

class Train(MemoryManager):
    LogTitle:str = 'ModelTrainer'
    """
    This class is to train models for other Input/Output processes.
    If you want to train your model, please write your module into train_modules in MasterConfig.
    
    This code calls 'modulename.Train()'. so you must write like this below.
        module_folder/__init__.py
        >>> from .train import Train
            
    """
    def __init__(self,device:torch.device,debug_mode:bool=False) -> None:
        """
        device [required] : torch.device
        debug_mode [Optional] : bool
        """
        super().__init__(self.LogTitle,debug_mode)
        self.module_length = len(Config.train_modules)
        self.debug_mode =debug_mode
        self.optional_debug_mode = [debug_mode] * self.module_length
        self.devices = [torch.device(device)] * self.module_length

    def activation(self,shutdown:mp.Value,sleep:mp.Value) -> None:
        """
        shutdown: multiprocessing shared memory bool value
        sleep   : multiprocessing shared memory bool value
        """
        modules = [importlib.import_module(m) for m in Config.train_modules]
        trains = []
        for (m,dev,deb) in zip(modules,self.devices,self.optional_debug_mode):
            trains.append(m.Train(dev,deb))
        trainlen = len(trains)
        logtitles = [i.LogTitle for i in trains]
        self.log('Trainer process has',*logtitles)
        trained = 0
        self.log(trains,debug_only=True)
        self.log('Instanced Train Modules')

        while not shutdown.value:
            if sleep.value:
                #for i in trains:
                #    self.log('training',i)
                #    i(shutdown,sleep)
                for _ in range(trainlen):
                    if not trained < trainlen:
                        trained = 0
                    self.log('Training',trains[trained].LogTitle)
                    try:
                        trains[trained](shutdown,sleep)
                    except RuntimeError:
                        if self.debug_mode:
                            self.exception(f'Runtime error ocurred!')
                        else:
                            self.warn(f'RuntimeError ocurred! Please check {trains[trained].LogTitle}.')
                    
                    self.release_system_memory()
                    if shutdown.value or not sleep.value:
                        break
                    time.sleep(Config.train_wait)
                self.log('training process ended')
                while sleep.value and not shutdown.value:
                    time.sleep(Config.train_wait)
            else:
                time.sleep(Config.train_wait)
        self.log('process shutdowned')
