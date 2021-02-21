from datetime import datetime
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
from .configure import config
from .DataProcess import DataEncoding
import torch
import cv2
from typing import Union,Tuple,Any
from os.path import isfile,isdir
from os.path import join as pathjoin
import h5py
import math
from .AutoEncoder import Encoder

class Sensation(MemoryManager):
    memory_format:str = '0'
    log_title:str = f'sensation{memory_format}'
    ReadOutLength:int = config.ReadoutLength
    MemoryListLength:int = config.MemListLength
    MemorySize:int = int(math.prod(Encoder().output_size))
    dtype:str = config.dtype

    def __init__(self,device:Union[str,torch.device],debug_mode:bool=False) -> None:
        super().__init__(self.log_title,debug_mode)
        self.device = torch.device(device)
        self.debug = Debug(self.log_title,debug_mode)
        if not isdir(config.temp_folder):
            self.debug.log('made',config.temp_folder)
            
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
        encoding = DataEncoding(self.log_title,self.device,dtype=config.torchdtype)
        self.debug.log('called DataEncoding')

        ## capture settings
        capture = cv2.VideoCapture(config.default_video_capture,cv2.CAP_DSHOW)
        if not capture.isOpened():
            self.debug.exception('cannot open video capture! Please check default_video_capture.')
        
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT,config.height)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH,config.width)
        capture.set(cv2.CAP_PROP_FPS,config.frame_rate)

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.debug.log('setted video capture')

        ## inherit shared memories
        ReadOutId = self.inherit_shared_memory(ReadOutId)
        ReadOutMemory = self.inherit_shared_memory(ReadOutMemory)
        memory_list = self.inherit_shared_memory(memory_list)

        ## load previous values
        if isfile(config.newestId_file):
            newest_id.value = self.load_python_obj(config.newestId_file)
            self.debug.log('loaded newest ID')
        else:
            newest_id.value = self.get_firstId(self.memory_format,return_integer=True)#-1
            ReadOutId[0] = newest_id.value
        
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
            current_length = 1#0
        saved_length = current_length

        ReadOutTime = np.full((self.ReadOutLength,),-1,dtype='float64')
        if isfile(config.ReadoutTime_file):
            ml = self.load_python_obj(config.ReadoutTime_file)[:current_length]
            ReadOutTime[:current_length] = ml
            self.debug.log('loaded ReadOutTime')
            
        
        ReadOutMemory_torch = torch.from_numpy(ReadOutMemory.copy()).to(self.device)
        
        ## Training data saver
        video_array = np.zeros((config.saving_rate,config.height,config.width,config.channels),dtype='uint8')
        saved_video_len = 0



        self.debug.log('process start!')
        while cmd.value != mconf.shutdown:

            clock_start = time.time()
            while not switch.value:
                if cmd.value == mconf.shutdown:
                    break
                clock.value = 0.0
                time.sleep(mconf.wait_time)
            
            ret,img = capture.read()
            if not ret:
                self.debug.exception('can not read image from capture!')
            cv2.imshow(self.log_title,img)

            data = encoding.encode(img)
            #self.debug.log(data.shape,ReadOutMemory_torch.shape)
            distances = encoding.calc_distance(data,ReadOutMemory_torch[:current_length])
            mins = torch.min(distances)
            if mins > 0.00001:
                ReadOutMemory[current_length] = data.to('cpu').numpy()
                ReadOutMemory_torch[current_length] = data
                newest_id.value += 1
                ReadOutId[current_length] = newest_id.value
                current_length += 1
                ### your process ------
                video_array[saved_video_len] = img
                saved_video_len += 1
                ### -------------------

            id_args = torch.argsort(distances)[:self.MemoryListLength].to('cpu').numpy()
            il = id_args.shape[0]
            memory_list[:il] = ReadOutId[id_args]

            if not (config.saving_rate > saved_video_len):
                """
                with h5py.File(config.video_data,'a') as f:
                    now_time = str(time.time())#datetime.now(mconf.TimeZone).strftime('%d-%m-%y_%H-%M-%S')
                    f.create_dataset(name=now_time,data=video_array)
                    saved_video_len = 0
                """
                name = pathjoin(config.data_folder,str(time.time()))
                self.save_python_obj(name,video_array)
                saved_video_len = 0
            

            if cmd.value == mconf.force_sleep or (not current_length < self.ReadOutLength) or (not switch.value):

                self.save_memory(
                    ReadOutId[saved_length:current_length].copy(),
                    ReadOutMemory[saved_length:current_length].copy(),
                    ReadOutTime[saved_length:current_length].copy(),
                )
                self.debug.log('saved memories')

                get_idx = np.random.permutation(current_length)[:config.KeepLength]
                get_idx = np.sort(get_idx)
                current_length = get_idx.shape[0]
                for mems in [ReadOutId,ReadOutMemory,ReadOutTime,ReadOutMemory_torch]:
                    mems[:current_length] = mems[get_idx]
                    mems[current_length:] = -1
                saved_length = current_length

                ## This place is saving place which you want to save python objects to file.
                self.save_python_obj(config.memlist_file,memory_list)
                self.save_python_obj(config.ReadoutTime_file,ReadOutTime[:current_length])
                self.save_python_obj(config.ReadoutId_file,ReadOutId[:current_length])
                self.save_python_obj(config.ReadoutMem_file,ReadOutMemory[:current_length])
                self.save_python_obj(config.newestId_file,newest_id.value)
                self.debug.log('saved Read Out Id,Memory,Time,memlist,newest_id')
                
                ## ------------------------------------------------------------
                if cmd.value == mconf.force_sleep:
                    time.sleep(mconf.sleep_wait)
            
            time.sleep(config.wait_time * sleep.value)
            cv2.waitKey(1)
            clock.value = time.time() - clock_start
        # -------- end while ---------------

        ### shutdown process
        self.debug.log('started shutdown process')
        capture.release()
        cv2.destroyAllWindows()
        self.save_memory(
                ReadOutId[saved_length:current_length].copy(),
                ReadOutMemory[saved_length:current_length].copy(),
                ReadOutTime[saved_length:current_length].copy(),
        )            

        self.save_python_obj(config.memlist_file,memory_list)
        self.save_python_obj(config.ReadoutTime_file,ReadOutTime[:current_length])
        self.save_python_obj(config.ReadoutId_file,ReadOutId[:current_length])
        self.save_python_obj(config.ReadoutMem_file,ReadOutMemory[:current_length])
        self.save_python_obj(config.newestId_file,newest_id.value)
        self.debug.log('saved Read Out Id,Memory,Time,memlist,newest_id')

        self.debug.log('process shutdowned')

