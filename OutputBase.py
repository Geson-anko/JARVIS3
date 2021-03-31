from MemoryManager import MemoryManager
from MasterConfig import Config
import time
import numpy as np
import torch
from typing import Union,Tuple,List
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory

class OutputBase(MemoryManager):
    LogTitle:str
    UsingMemoryFormat:str
    MaxMemoryLength:int
    UseMemoryLowerLimit:int = 1
    
    SleepWaitTime:float
    MaxFrameRate:int = 100

    dtype:np.dtype
    torchdtype:torch.dtype

    def __init__(self, device:Union[str,torch.device], debug_mode: bool=False) -> None:
        super().__init__(self.LogTitle,debug_mode)
        self.device = torch.device(device)

    @torch.no_grad()
    def activation(
        self,shutdown:mp.Value,sleep:mp.Value,switch:mp.Value,clock:mp.Value,sleepiness:mp.Value,
        TemporaryMemory:Tuple[np.ndarray,SharedMemory],
        ReadOutIds:List[Tuple[np.ndarray,SharedMemory]],
        ReadOutMemories:List[Tuple[np.ndarray,SharedMemory]],
        IsActives:List[mp.Value],
        ) -> None:
        """
        shutdown        : multiprocessing shared memory bool value.
        sleep           : multiprocessing shared memory bool value.
        switch          : multiprocessing shared memory bool value.
        clock           : multiprocessing shared memory dubble value.
        sleepiness      : multiprocessing shared memory dubble value.
        ReadOutIds      : list of shared memory objects
        ReadOutMemories : list of shared memory objects
        IsActives        : list of multiprocessing shared memory bool value.
        """
        self.shutdown = shutdown
        self.sleep = sleep
        self.switch = switch
        self.clock = clock
        self.sleepiness = sleepiness
        self.TemporaryMemory = TemporaryMemory
        self.ReadOutId = ReadOutIds
        self.ReadOutMemory = ReadOutMemories
        self.IsActives = IsActives

        self.LoadModels()
        self.InheritSharedMemories()
        self.WaitAllTrue()
        self.SetUsingReadOuts()
        self.Start() # start process

        # ------ Output Process ------
        self.log('process start')
        system_cache_time = time.time()
        first_call = True
        self.release_system_memory()
        while not self.shutdown.value:
            clock_start = time.time()
            self.SwitchCheck()

            memory = self.GetMemory()
            if memory.size(0) >= self.UseMemoryLowerLimit:
                if first_call:
                    self.log('called')
                    first_call = False
                self.Update(memory)

            
            if Config.release_system_cache_time < (time.time() - system_cache_time):
                self.release_system_memory()
                system_cache_time = time.time()

            time.sleep(self.SleepWaitTime*self.sleepiness.value)
            if self.sleep.value:
                self.SleepProcess()

            self.UpdateEnd()
            # set frame rate
            wait = (1/self.MaxFrameRate) - (time.time() - clock_start)
            if wait > 0:
                time.sleep(wait)
            
            self.clock.value = (time.time() - clock_start)
        
        # ------- end while ------
        self.log('started shutdown process')
        self.End()
        self.log('process shutdowned')        

    def InheritSharedMemories(self) -> None:
        self.TemporaryMemory = self.inherit_shared_memory(self.TemporaryMemory)
        self.ReadOutId = [self.inherit_shared_memory(i) for i in self.ReadOutId]
        self.ReadOutMemory = [self.inherit_shared_memory(i) for i in self.ReadOutMemory]
        self.log('Inherit Shared Memories')

    def WaitAllTrue(self) -> None:
        self.log('waiting until all sensations are Active.')
        while not self.shutdown.value:
            alltrue = True
            for i in self.IsActives:
                alltrue *= i.value
            if alltrue:
                break
            else:
                time.sleep(Config.all_true_wait_time)
        self.log('finished WaitAllTrue')

    def SetUsingReadOuts(self) ->None:
        UsingMemory,UsingId = None,None
        for idx,roi in enumerate(self.ReadOutId):
            i = self.Num2Id(roi[0])[0]
            if i == self.UsingMemoryFormat:
                UsingMemory,UsingId = self.ReadOutMemory[idx],self.ReadOutId[idx]
                break
        if UsingId is None:
            self.exception(f'You will use MemoryFormat {self.UsingMemoryFormat}, but not existing in ReadOutMemories.')
        
        self.ReadOutId = UsingId
        self.ReadOutMemory = UsingMemory

        self.log('Found ReadOutMemory and Ids')

    def GetMemory(self) -> torch.Tensor:
        cid = self.TemporaryMemory.copy()
        useid = self.extract_sameId(cid[cid!=Config.init_id],self.UsingMemoryFormat,return_integer=True)
        np.random.shuffle(useid)
        useid = useid[:self.MaxMemoryLength]
        memory = self.load_memory_withReadOuts(useid,self.ReadOutId,self.ReadOutMemory)
        return torch.from_numpy(memory.copy()).type(self.torchdtype).to(self.device)

    def SwitchCheck(self) -> None:
        if not self.switch.value:
            self.log('switch off.')
            while not self.switch.value:
                if self.shutdown.value:
                    break
                self.clock.value = 0.0
                time.sleep(Config.wait_time)
            self.log('switch on')

    def SleepProcess(self) -> None:
        time.sleep(Config.sleep_wait)
        self.LoadModels()
    
    def LoadModels(self) -> None:pass
    def Start(self) -> None:pass
    def Update(self,MemoryData:torch.Tensor) -> None:pass
    def UpdateEnd(self) -> None:pass
    def End(self) -> None:pass
        
        
        