from MemoryManager import MemoryManager
from MasterConfig import Config

from torch.nn import Module
import torch
from torch import Tensor
import numpy as np
from typing import Union,Tuple
import os
from os.path import isdir,isfile
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
import time

class SensationBase(MemoryManager):
    MemoryFormat:str
    LogTitle:str
    ReadOutLength:int
    KeepLength:int
    MemoryListLength:int
    MemorySize:int
    DataSize:int
    DataSavingRate:int
    
    Encoder:Module
    DeltaTime:Module
    SleepWaitTime:float

    Current_directory:str
    Param_folder:str
    Data_folder:str
    Temp_folder:str

    ## defining parameter file
    Encoder_params:str
    Decoder_params:str
    DeltaTime_params:str

    ## defining temporary file
    NewestId_file:str
    MemoryList_file:str
    ReadOutMemory_file:str
    ReadOutId_file:str
    ReadOutTime_file:str

    dtype:np.dtype
    torchdtype:torch.dtype


    def __init__(self, device:Union[str,torch.device], debug_mode: bool) -> None:
        super().__init__(log_title=self.LogTitle, debug_mode=debug_mode)
        self.device = torch.device(device)
        if not isdir(self.Temp_folder):
            os.makedirs(self.Temp_folder)
            self.log('made',self.Temp_folder)
        if not isdir(self.Data_folder):
            os.makedirs(self.Data_folder)
            self.log('made',self.Data_folder)
        if not isdir(self.Param_folder):
            self.exception('Not exist Param_folder')

    def activation(
        self,shutdown:mp.Value,sleep:mp.Value,switch:mp.Value,clock:mp.Value,sleepiness:mp.Value,
        ReadOutId:Tuple[np.ndarray,SharedMemory],
        ReadOutMemory:Tuple[np.ndarray,SharedMemory],
        MemoryList:Tuple[np.ndarray,SharedMemory],
        NewestId:mp.Value,
        IsActive:mp.Value,
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
        IsActive: multiprocessing shared memory bool value.
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

        self.DataArray = np.zeros((self.DataSavingRate,self.DataSize),dtype=self.dtype)
        self.SavedDataLen = 0

        self.LoadEncoder()
        self.LoadDeltaTime()
        self.InheritSharedMemories()
        self.LoadPreviousValues()

        self.Start() # start process
        # ------ Sensation Process ------
        self.log('process start')
        IsActive.value = True
        while not self.shutdown.value:
            clock_start = time.time()
            self.SwitchCheck()

            self.Data = self.Update() # update process
            self.MemoryProcess()
            if self.sleep.value or (not self.current_length < self.ReadOutLength) or (not self.switch.value):
                self.SaveMemories()
                self.SavePreviousValues()
            
            if self.sleep.value:
                time.sleep(Config.sleep_wait)
                self.LoadEncoder()
                self.LoadDeltaTime()

            #sleepiness wait
            time.sleep(self.SleepWaitTime*self.sleepiness.value)
            self.UpdateEnd()
            self.clock.value = time.time() - clock_start
        # ------ end while ------
        IsActive.value = False
        self.log('started shutdown process')
        self.SaveMemories()
        self.SavePreviousValues()
        self.End()
        
        self.log('process shutdowned')

    def LoadEncoder(self) -> None:
        self.encoder = self.Encoder().to(self.device).type(self.torchdtype)
        self.encoder.load_state_dict(torch.load(self.Encoder_params,map_location=self.device))
        self.log('loaded Encoder')
    
    def LoadDeltaTime(self) -> None:
        self.deltaT = self.DeltaTime().to(self.device).type(self.torchdtype)
        self.deltaT.load_state_dict(torch.load(self.DeltaTime_params,map_location=self.device))
        self.log('loaded DeltaT')

    def InheritSharedMemories(self) -> None:
        self.ReadOutId = self.inherit_shared_memory(self.ReadOutId)
        self.ReadOutMemory = self.inherit_shared_memory(self.ReadOutMemory)
        self.MemoryList = self.inherit_shared_memory(self.MemoryList)
        self.log('Inherit Shared Memories')

    def LoadPreviousValues(self) -> None:
        """
        Loading previous NewestId,MemoryList,ReadOutId,ReadOutMemory and ReadOutTime.
        And create ReadOutMemory_torch. 
        """
        # load NewestId
        if isfile(self.NewestId_file):
            self.NewestId.value = self.load_python_obj(self.NewestId_file)
            self.log('loaded NewestId')
        else:
            self.NewestId.value = self.get_firstId(self.memory_format,return_integer=True)
            self.ReadOutId[0] = self.NewestId.value
            self.log('got firstId')

        # load MemoryList
        if isfile(self.MemoryList_file):
            ml = self.load_python_obj(self.MemoryList_file)
            ml = ml[:self.MemoryListLength]
            mlen = ml.shape[0]
            self.MemoryList[:mlen] = ml
            self.log('loaded MemoryList')
 
        # load ReadOutMemory ,Id and Time
        self.ReadOutTime = np.zeros((self.ReadOutLength,),dtype='float64')
        if isfile(self.ReadOutId_file):
            ml = self.load_python_obj(self.ReadOutId_file)[:self.ReadOutLength]
            if isfile(self.ReadOutMemory_file) and isfile(self.ReadOutTime_file):
                # load id
                current_length = ml.shape[0]
                self.ReadOutId[:current_length]= ml
                # load memory
                ml = self.load_python_obj(self.ReadOutMemory_file)[:current_length]
                self.ReadOutMemory[:current_length] = ml
                # load time
                ml = self.load_python_obj(self.ReadOutTime_file)[:current_length]
                self.ReadOutTime[:current_length] = ml

                self.log('loaded ReadOutMemory, Id and Time.')
            else:
                self.warn(f'ReadOutMemory_file ->{isfile(self.ReadOutMemory_file)}, ReadOutTime_file -> {isfile(self.ReadOutTime_file)} do not exist. so load memories from memories file.')
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
                time.sleep(Config.wait_time)
            self.log('switch on')

    @torch.no_grad()
    def MemoryProcess(self) -> None:
        """
        
        """
        if type(self.Data) is not torch.Tensor:
            self.exception(f'TypeError from MemoryProcess. Input data type is {type(self.Data)}')
        encoded = self.encoder(self.Data.type(self.torchdtype)).view(-1)
        data = encoded.unsqueeze(0).repeat(self.current_length,1)
        distances = self.deltaT(data,self.ReadOutMemory_torch[:self.current_length])
        mins = torch.min(distances)
        if mins > Config.deltaT_threshold:
            self.ReadOutMemory[self.current_length] = encoded.to('cpu').numpy()
            self.ReadOutMemory_torch[self.current_length] = encoded
            self.NewestId.value += 1
            self.ReadOutId[self.current_length] = self.NewestId.value
            self.current_length += 1
            # data saving -----
            self.DataArray[self.SavedDataLen] = self.Data.detach().numpy()
            self.SavedDataLen +=1
            #------------------
        id_args = torch.argsort(distances)[:self.MemoryListLength].to('cpu').numpy()
        il = id_args.shape[0]
        self.MemoryList[:il] = self.ReadOutId[id_args]

        #DataSavingCheck
        if not(self.DataSavingRate > self.SavedDataLen):
            name = os.path.join(self.Data_folder,str(time.time()))
            self.save_python_obj(name,self.DataArray)
            self.SavedDataLen = 0

    def SaveMemories(self) -> None:
        """
        Saving ReadOutId,ReadOutMemory,ReadOutTime to memory file
        """
        self.save_memory(
            self.ReadOutId[self.saved_length:self.current_length].copy(),
            self.ReadOutMemory[self.saved_length:self.current_length].copy(),
            self.ReadOutTime[self.saved_length:self.current_length].copy(),
        )
        self.saved_length = self.current_length
        self.log('Saved memories')

        # Reducing Memories
        get_idx = np.random.permutation(self.current_length)[:self.KeepLength]
        get_idx = np.sort(get_idx)
        self.current_length = get_idx.shape[0]
        for mems in [self.ReadOutId,self.ReadOutMemory,self.ReadOutTime,self.ReadOutMemory_torch]:
            mems[:self.current_length] = mems[get_idx]
            mems[self.current_length:] = Config.init_id
        self.saved_length = self.current_length
        self.log('Reduced memories')

    def SavePreviousValues(self) -> None:
        """
        Saving previous NewestId,MemoryList,ReadOutId,ReadOutMemory and ReadOutTime.
        """
        self.save_python_obj(self.MemoryList_file,self.MemoryList)
        self.save_python_obj(self.ReadOutMemory_file,self.ReadOutMemory[:self.current_length])
        self.save_python_obj(self.ReadOutId_file,self.ReadOutId[:self.current_length])
        self.save_python_obj(self.ReadOutTime_file,self.ReadOutTime[self.current_length])
        self.save_python_obj(self.NewestId_file,self.NewestId.value)
        self.log('saved Read Out Id,Memory,Time,memlist,NewestId')
        

        
    def Start(self):pass
    def Update(self):pass
    def UpdateEnd(self):pass
    def End(self):pass
