import time
import os
import sys
from multiprocessing.shared_memory import SharedMemory
import multiprocessing as mp
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from debug_tools import Debug
from MemoryManager import MemoryManager
from MasterConfig import Config as mconf
from .config import config
from .DataProcess import DataEncoding
import torch
import cv2
from typing import Union,Tuple,Any
from os.path import isfile
import h5py

class Sensation(MemoryManager):
    memory_format:str = '0'
    log_title:str = f'sensation{memory_format}'
    ReadOutLength:int = config.ReadoutLength
    MemoryListLength:int = config.MemListLength
    MemorySize:int = config.a_memory_size

    def __init__(self,device:Union[str,torch.device],debug_mode:bool=False) -> None:
        super().__init__(self.log_title,debug_mode)
        self.device = device
        self.debug = Debug(self.log_title,debug_mode)
        
    def activation(
        self,cmd:mp.Value,switch:mp.Value,clock:mp.Value,sleep:mp.Value,
        ReadOutId:Tuple[np.ndarray,SharedMemory],
        ReadOutMemory:Tuple[np.ndarray,SharedMemory],
        memory_list:Tuple[np.ndarray,SharedMemory],
        newest_id:mp.Value,
        ) -> None:
        """
        cmd:    multiprocessing shared memory int value.
        switch: multiprocessing shared memory bool value.
        clock:  multiprocessing shared memory dubble value.
        sleep:  multiprocessing shared memory dubble value.
        ReadOutId:  shared memory objects
        ReadOutmemory:  shared memory objects
        memory_list:    shared memory objects
        newest_id:  mutiprocessing shared memory int value.
        """

        ## call functions
        encoding = DataEncoding(self.log_title,self.device)
        self.debug.log('called DataEncoding')

        ## capture settings
        capture = cv2.VideoCapture(config.default_video_capture)
        if not capture.isOpened():
            self.debug.exception('cannot open video capture! Please check default_video_capture.')
        
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,config.height)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,config.width)
        capture.set(cv2.CAP_PROP_FPS,config.frame_rate)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT,config.height))

        self.debug.log('setted video capture')

        ## inherit shared memories
        ReadOutId = self.inherit_shared_memory(ReadOutId)
        ReadOutMemory = self.inherit_shared_memory(ReadOutMemory)
        memory_list = self.inherit_shared_memory(memory_list)

        ## load previous values
        if isfile(config.newestId_file):
            newest_id.value = self.load_python_obj(config.newestId_file)
        else:
            newest_id.value = -1 #self.get_firstId(self.memory_format,return_integer=True)
        self.debug.log('loaded newest ID')

        if isfile(config.memlist_file):
            ml = self.load_python_obj(config.memlist_file)
            ml = ml[:self.MemoryListLength]
            mlen = ml.shape[0]
            memory_list[:mlen] = ml
            self.debug.log('loaded memory list')


        if isfile(config.ReadoutId_file):
            ml = self.load_python_obj(config.ReadoutId_file)[:self.ReadOutLength]
            if isfile(config.ReadoutMem_file):
                current_length = ml.shape[0] # id load
                ReadOutId[:current_length] = ml # id load
                
                # memory load
                ml = self.load_python_obj(config.ReadoutMem_file)[:current_length]
                ReadOutMemory[:current_length] = ml
            else:
                ml,rm = self.load_memory(ml,return_id_int=True)
                current_length = rm.shape[0]
                ReadOutMemory[:current_length] = rm
                ReadOutId[:current_length] = ml
            self.debug.log(f'loaded ReadOutMemory and ReadOutId. current length is {current_length}')
        else:
            current_length = 0
        saved_length = current_length

        ReadOutTime = np.full((self.ReadOutLength,),-1,dtype='float64')
        if isfile(config.ReadoutTime_file):
            ml = self.load_python_obj(config.ReadoutTime_file)[:current_length]
            ReadOutTime[:current_length] = ml
            self.debug.log('loaded ReadOutTime')
            
        
        ReadOutMemory_torch = torch.from_numpy(ReadOutMemory.copy()).to(self.device)
        
        ## Training data saver
        #video_writer = h5py.File()


        self.debug.log('process start!')
        while cmd.value != mconf.shutdown:

            clock_start = time.time()
            while not switch.value:
                clock.value = 0.0
                time.sleep(mconf.wait_time)
            
            ret,img = capture.read()
            if not ret:
                self.debug.exception('can not read image from capture!')
            
            data = encoding.encode(img)
            distances = encoding.calc_distance(data,ReadOutMemory_torch[:current_length])
            mins = torch.min(distances)
            if mins > 0.00001:
                ReadOutMemory[current_length] = data.to('cpu').numpy()
                ReadOutMemory_torch[current_length] = data
                newest_id.value += 1
                ReadOutId[current_length] = newest_id.value
                current_length += 1

            id_args = torch.argsort(distances)[:self.MemoryListLength].to('cpu').numpy()
            il = id_args.shape[0]
            memory_list[:il] = ReadOutId[id_args]

            if cmd.value == mconf.force_sleep or (not current_length < self.ReadOutLength):
                self.save_memory(
                    ReadOutId[saved_length:current_length],
                    ReadOutMemory[saved_length:current_length],
                    ReadOutTime[saved_length:current_length],
                )
                get_idx = np.random.permutation(current_length)[:config.KeepLength]
                for mems in [ReadOutId,ReadOutMemory,ReadOutTime,ReadOutMemory_torch]:
                    mems[:config.KeepLength] = mems[get_idx]
                    mems[config.KeepLength:] = -1
                self.debug.log('saved memories')
                saved_length = config.KeepLength
                current_length = saved_length
            
            time.sleep(config.wait_time * sleep.value)
            cv2.waitKey(1)
            clock.value = time.time() - clock_start

        ### shutdown process
        self.debug.log('started shutdown process')
        self.save_memory(
                ReadOutId[saved_length:current_length],
                ReadOutMemory[saved_length:current_length],
                ReadOutTime[saved_length:current_length],
        )            

        self.save_python_obj(config.memlist_file,memory_list)
        self.save_python_obj(config.ReadoutTime_file,ReadOutTime[:current_length])
        self.save_python_obj(config.ReadoutId_file,ReadOutId[:current_length])
        self.save_python_obj(config.ReadoutMem_file,ReadOutMemory[:current_length])
        self.save_python_obj(config.newestId_file,newest_id.value)
        self.debug.log('saved Read Out Id,Memory,Time,memlist,newest_id')

        self.debug.log('process shutdowned')
        
