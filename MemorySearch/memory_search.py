import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from MasterConfig import Config as mconf
from MemoryManager import MemoryManager
from .config import config

import numpy as np
import multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from typing import Tuple,Union,TypeVar
from os.path import isfile
import copy
import time

ID_list_type = TypeVar('ID_list_type')
MemDict_type = TypeVar('MemDict_type')

class MemorySearch(MemoryManager):
    LogTitle:str = 'MemorySearch'

    def __init__(self,debug_mode:bool = False):
        super().__init__(self.LogTitle,debug_mode)

    def activation(
        self,
        shutdown:mp.Value,
        sleep:mp.Value,
        switch:mp.Value,
        clock:mp.Value,
        sleepiness:mp.Value,
        TempMemory:Tuple[np.ndarray,SharedMemory],
        mem_lists:Tuple[Tuple[np.ndarray,SharedMemory],...],
        newest_ids:Tuple[mp.Value,...],
    ) -> None:
        """
        shutdown:    multiprocessing shared memory bool value.
        sleep:    multiprocessing shared memory bool value.
        clock:  multiprocessing shared memory dubble value.
        sleepiness:  multiprocessing shared memory dubble value.
        TempMemory: shared memory objects
        memory_lists: Tuple of shared memory objects
        newest_ids: Tuple of mutiprocessing shared memory long values
        """
        # inherit shared memory
        self.switch = switch
        self.shutdown = shutdown
        self.clock = clock
        self.sleepiness = sleepiness
        
        mem_lists = [self.inherit_shared_memory(i) for i in mem_lists]
        TempMemory = self.inherit_shared_memory(TempMemory)

        # load file
        if isfile(config.dict_file):
            MemoryDict = self.load_python_obj(config.dict_file)
            self.log('loaded memory dictionary')
        else:
            MemoryDict = dict()
            self.log('create new memory dictionary')

        templen = TempMemory.shape[0]
        if isfile(config.tempmem_file):
            _m = self.load_python_obj(config.tempmem_file)[:templen]
            _l = _m.shape[0]
            TempMemory[:_l] = _m
            self.log('loaded Temporary Memory')

        # set default value
        old_memlist = [copy.deepcopy(i) for i in mem_lists]
        old_ids = [i.value for i in newest_ids]

        self.log('Temporary Memory Length:',templen)
        self.log('Memory Dictionary Length:',len(MemoryDict))

        # process --------------------------------------
        self.log('process start')
        while not shutdown.value:
            clock_start = time.time()
            self.SwitchCheck()
            time.sleep(sleepiness.value * config.wait_time)
            if sleep.value:
                time.sleep(mconf.sleep_wait)
                self.save_python_obj(config.tempmem_file,TempMemory.copy())
                self.save_python_obj(config.dict_file,MemoryDict)

            # modified check
            new_memlist = [i.copy() for i in mem_lists]
            modified_memlist = [i if self.mem_list_modified(i,q) else [] for i,q in zip(new_memlist,old_memlist)]
            old_memlist = new_memlist
            # Update Temporary Memory
            c_tmp = TempMemory.copy()
            c_tmp = np.concatenate([c_tmp,*modified_memlist])
            c_tmp = c_tmp[c_tmp != mconf.init_id]
            c_tmp = np.unique(c_tmp)
            np.random.shuffle(c_tmp)
            c_tmp = c_tmp[:templen]
            searched = self.Search(c_tmp,MemoryDict)
            c_tmp = np.unique(np.concatenate([c_tmp,searched]))
            np.random.shuffle(c_tmp)
            c_len = c_tmp.shape[0]
            TempMemory[:c_len] = c_tmp

            ## Update Memory Dictionary
            # new id connection
            _newids = [i.value for i in newest_ids]
            modified_ids = [i for i,q in zip(_newids,old_ids) if i != q]
            old_ids = _newids
            idxes = np.arange(-config.max_connection,c_len)
            idxes[:config.max_connection][idxes[:config.max_connection] < -c_len] = -c_len
            np.random.shuffle(idxes)
            for i,nid in enumerate(modified_ids):
                MemoryDict[nid] = c_tmp[i:i+config.max_connection]
            
            np.random.shuffle(idxes)    
            # reconnection
            for i,nid in enumerate(c_tmp):
                MemoryDict[nid] = c_tmp[i:i+config.max_connection]
            
            time.sleep(sleepiness.value * config.wait_time)
            if sleep.value:
                self.save_python_obj(config.tempmem_file,TempMemory.copy())
                self.save_python_obj(config.dict_file,MemoryDict)

            wait = (1/config.max_frame_rate) - (time.time() - clock_start)
            if wait > 0:
                time.sleep(wait)
            clock.value = time.time() - clock_start
            #print(c_tmp)
            #raise Exception('stop')

        # shutdown process --------------------------------------------------
        self.log('shutdown process started')
        self.save_python_obj(config.tempmem_file,TempMemory.copy())
        self.save_python_obj(config.dict_file,MemoryDict)
        self.log('saved Memory Dictionary and Temporary Memory')
        self.log('memorydict',MemoryDict,debug_only=True)
        self.log('TempMem',TempMemory,debug_only=True)
        self.log('process shutdowned')

    
    def mem_list_modified(self,new:np.ndarray,old:np.ndarray) -> bool:
        modified = False
        for (p,q) in zip(new,old):
            if p!= q:
                modified = True
                break
        return modified

    def Search(self,ID_list:ID_list_type,mem_dict:MemDict_type) -> ID_list_type:
        self.searched_id = [[]]
        for i in ID_list:
            try:
                self.searched_id.append(mem_dict[i])
            except KeyError:
                pass
        return np.concatenate(self.searched_id)
        
    def SwitchCheck(self) -> None:
        if not self.switch.value:
            self.log('switch off.')
            while not self.switch.value:
                if self.shutdown.value:
                    break
                self.clock.value = 0.0
                time.sleep(mconf.wait_time)
            self.log('switch on')
        
            
            

            
