import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
import torch
import numpy as np
from os.path import isfile,isdir, join as pathjoin
import time

from MemoryManager import MemoryManager
from MasterConfig import Config as mconf
from .config import config
from .sensation_models import Encoder,DeltaT

from typing import Union,Tuple

class Sensation(MemoryManager):
    memory_format:str = '_' # your process memory format (id[0])
    log_title:str = f'sensation{memory_format}'
    ReadOutLength:int = config.ReadOutLength
    MemoryListLength:int = config.MemoryListLength
    MemorySize:int = int(np.prod(Encoder().output_size))
    dtype:str = config.dtype

    def __init__(self, device:Union[str,torch.device], debug_mode: bool) -> None:
        super().__init__(log_title=self.log_title, debug_mode=debug_mode)
        self.device = torch.device(device)
        if not isdir(config.temp_folder):
            os.mkdir(config.temp_folder)
            self.log('made',config.temp_folder)
            
    def activation(
        self,shutdown:mp.Value,sleep:mp.Value,switch:mp.Value,clock:mp.Value,sleepiness:mp.Value,
        ReadOutId:Tuple[np.ndarray,SharedMemory],
        ReadOutMemory:Tuple[np.ndarray,SharedMemory],
        MemoryList:Tuple[np.ndarray,SharedMemory],
        NewestId:mp.Value,
        ) -> None:
        """
        shutdown:   multiprocessing shared memory bool value.
        sleep:  multiprocessing shared memory bool value.
        switch: multiprocessing shared memory bool value.
        clock:  multiprocessing shared memory dubble value.
        sleepiness:  multiprocessing shared memory dubble value.
        ReadOutId:  shared memory objects
        ReadOutmemory:  shared memory objects
        MemoryList:    shared memory objects
        NewestId:  mutiprocessing shared memory int value.
        """
        self.shutdown = shutdown
        self.sleep = sleep
        self.switch = switch
        self.clock = clock
        self.sleepiness = sleepiness
        self.ReadOutId = ReadOutId
        self.ReadOutMemory = ReadOutMemory
        self.MemoryList = MemoryList
        self.NewestId = NewestId
        self.LoadEncoder()
        self.LoadDeltaT()
        self.InheritSharedMemories()
        self.LoadPreviousValues()

        # ------ your start process ------

        # ------ end start process ------
        
        # ------ Sensation Process ------
        self.log('process start')
        while not self.shutdown.value:
            clock_start = time.time()
            self.SwitchCheck()

            # ------ your update process ------
            
            self.data = "preprocessed input data, it is torch.Tensor"
            # ------ end update process ------




    def LoadEncoder(self) -> None:
        self.encoder = Encoder().to(self.device)
        self.encoder.load_state_dict(torch.load(config.encoder_params,map_location=self.device))
        self.log('loaded Encoder')

    def LoadDeltaT(self) -> None:
        self.deltaT = DeltaT().to(self.device)
        self.deltaT.load_state_dict(torch.load(config.deltatime_params,map_location=self.device))
        self.log('loaded DeltaT')

    def InheritSharedMemories(self) -> None:
        self.ReadOutId = self.inherit_shared_memory(self.ReadOutId)
        self.ReadOutMemory = self.inherit_shared_memory(self.ReadOutMemory)
        self.MemoryList = self.inherit_shared_memory(self.MemoryList)
        self.log('Inherit Shared Memories')

    def LoadPreviousValues(self) -> None:
        # load NewestId
        if isfile(config.NewestId_file):
            self.NewestId.value = self.load_python_obj(config.NewestId_file)
            self.log('loaded NewestId')
        else:
            self.NewestId.value = self.get_firstId(self.memory_format,return_integer=True)
            self.ReadOutId[0] = self.NewestId.value
            self.log('got firstId')

        # load MemoryList
        if isfile(config.MemoryList_file):
            ml = self.load_python_obj(config.MemoryList_file)
            ml = ml[:self.MemoryListLength]
            mlen = ml.shape[0]
            self.MemoryList[:mlen] = ml
            self.log('loaded MemoryList')
 
        # load ReadOutMemory ,Id and Time
        self.ReadOutTime = np.zeros((self.ReadOutLength,),dtype='float64')
        if isfile(config.ReadOutId_file):
            ml = self.load_python_obj(config.ReadOutId_file)[:self.ReadOutLength]
            if isfile(config.ReadOutMemory_file) and isfile(config.ReadOutTime_file):
                # load id
                current_length = ml.shape[0]
                self.ReadOutId[:current_length]= ml
                # load memory
                ml = self.load_python_obj(config.ReadOutMemory_file)[:current_length]
                self.ReadOutMemory[:current_length] = ml
                # load time
                ml = self.load_python_obj(config.ReadOutTime_file)[:current_length]
                self.ReadOutTime[:current_length] = ml

                self.log('loaded ReadOutMemory, Id and Time.')
            else:
                self.warn(f'ReadOutMemory_file ->{isfile(config.ReadOutMemory_file)}, ReadOutTime_file -> {isfile(config.ReadOutTime_file)} do not exist. so load memories from memories file.')
                ml,rm,tm = self.load_memory(ml,return_id_int=True,return_time=True)
                current_length = rm.shape[0]
                self.ReadOutMemory[:current_length] = rm
                self.ReadOutId[:current_length] = ml
                self.ReadOutTime[:current_length] = tm
                self.log('ReadOutMemory, Id and Time are loaded safely.')
            self.log(f'current memory length is {current_length}')
        else:
            current_length = 1
        self.current_length = current_length
        self.saved_length = current_length
        self.ReadOutMemory_torch = torch.from_numpy(self.ReadOutMemory.copy()).to(self.device)

    def SwitchCheck(self) -> None:
        if not self.switch.value:
            self.log('switch off.')
            while not self.switch.value:
                if self.shutdown.value:
                    break
                self.clock.value = 0.0
                time.sleep(mconf.wait_time)
            self.log('switch on')

    @torch.no_grad()
    def MemoryProcess(self) -> None:
        if type(self.data) is not torch.Tensor:
            self.exception(f'TypeError from MemoryProcess. Input data type is {type(self.data)}')
        encoded = self.encoder(self.data).view(-1)
        data = encoded.unsqueeze(0).repeat(self.current_length,1)
        distances = self.deltaT(data,self.ReadOutMemory_torch[:self.current_length])
        mins = torch.min(distances)
        if mins > mconf.deltaT_threshold:
            self.ReadOutMemory[self.current_length] = encoded.to('cpu').numpy()
            self.ReadOutMemory_torch[self.current_length] = encoded
            self.NewestId.value += 1
            self.ReadOutId[self.current_length] = self.NewestId.value
            self.CameNew= True
            self.current_length += 1
        else:
            self.CameNew = False
        id_args = torch.argsort(distances)[:self.MemoryListLength].to('cpu').numpy()
        il = id_args.shape[0]
        self.MemoryList[:il] = self.ReadOutId[id_args]

            


""" Documentation
This code is your sensation process.
"""