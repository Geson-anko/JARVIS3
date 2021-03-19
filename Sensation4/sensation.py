import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import math
import torch
from torch.nn import Module
from os.path import join as pathjoin

from .config import config
from SensationBase import SensationBase
from .sensation_models import Encoder

from pynput import mouse
import numpy as np
import time
from screeninfo import get_monitors

class Sensation(SensationBase):
    MemoryFormat:str = '4'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = 16384 # ReadOutLength
    KeepLength:int = math.floor(ReadOutLength*0.7)  # ReadOutLength * 0.7
    MemoryListLength:int = 16# 1% of ReadOutLength
    MemorySize:int = int(np.prod(Encoder.output_size))
    SameThreshold:float = 0.001 # The Threshold of memory error.
    DataSize:tuple = Encoder.input_size[1:]
    DataSavingRate:int = 2

    Encoder:Module = Encoder
    SleepWaitTime:float = 0.1

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    Data_folder:str = pathjoin(Current_directory,'data') # /current_dicrectory/data/
    Temp_folder:str = pathjoin(Current_directory,'temp') # /Current_directory/temp/

    ## defining parameter file
    Encoder_params:str= pathjoin(Param_folder,'encoder.params') # your encoder parameter file name
    Decoder_params:str = pathjoin(Param_folder,'decoder.params') # yout decoder parameter file name

    ## defining temporary file
    NewestId_file:str = pathjoin(Temp_folder,'NewestId.pkl')
    MemoryList_file:str = pathjoin(Temp_folder,'MemoryList.pkl')
    ReadOutMemory_file:str = pathjoin(Temp_folder,'ReadOutMemory.pkl')
    ReadOutId_file:str = pathjoin(Temp_folder,'ReadOutId.pkl')
    ReadOutTime_file:str = pathjoin(Temp_folder,'ReadOutTime.pkl')

    dtype:np.dtype = np.float32  # data type
    torchdtype:torch.dtype = torch.float32 # torch tensor data type

    # ------ your settings ------
    monitor = get_monitors()[0]
    monitor_width = monitor.width
    monitor_height = monitor.height
    frame_rate = 30
    frame_second = 1/frame_rate
    elapsed = frame_second
    move_x,move_y,scroll_dx,scroll_dy = 0,0,0,0

    def Start(self) -> None:
        # This method is called when process start.
        self.buttons = [*mouse.Button]
        self.buttons_array = np.zeros(len(self.buttons),dtype=bool)
        self.canvas = np.zeros(config.mouse_elems,self.dtype)
        self.mouse_listener = mouse.Listener(
            on_move=self.move,
            on_click=self.click,
            on_scroll=self.scroll,
        )
        self.mouse_listener.start()
        self.old_x,self.old_y = self.move_x/self.monitor_width,self.move_y/self.monitor_height

    def Update(self) -> torch.Tensor:
        # This method is called every frame start.
        # your data process
        self.start = time.time()
        now_x,now_y = self.move_x / self.monitor_width,self.move_y/self.monitor_height
        dx = (now_x - self.old_x) / self.elapsed
        dy = (now_y - self.old_y) / self.elapsed
        self.old_x,self.old_y = now_x,now_y
        self.canvas[0] = now_x
        self.canvas[1] = now_y
        self.canvas[2] = dx
        self.canvas[3] = dy
        self.canvas[4] = self.scroll_dx
        self.canvas[5] = self.scroll_dy
        self.canvas[6:] = self.buttons_array
        #print('\rcurrent_length',self.current_length,'mins:',self.mins,f'fps:{1/self.elapsed:3.2f}',end='')
        #basetext = '{0:2.2f}'
        #text = ' |'.join([basetext.format(i) for i in self.canvas])
        #print('\r',text,end='')

        return torch.from_numpy(self.canvas.copy())

    def UpdateEnd(self) -> None:
        # This method is called every frame end.
        wait = self.frame_second - (time.time() - self.start)
        if wait > 0:
            time.sleep(wait)
        self.elapsed = time.time() - self.start

    def End(self) -> None:
        # This method is called when shutting down process.
        pass

    def move(self,x,y):
        self.move_x,self.move_y = x,y
    
    def click(self,x,y,button,pressed):
        if not button in self.buttons:
            return
        self.buttons_array[self.buttons.index(button)] = pressed
    
    def scroll(self,x,y,dx,dy):
        self.scroll_dy,self.scroll_dx = dy,dx
        time.sleep(self.frame_second)
        self.scroll_dx,self.scroll_dy = 0,0

    def DataSavingCheck(self) -> None:
        self.SavedDataLen = 0
    # ------ train settings ------
    Training_dtype:torch.dtype = torch.float16
    AutoEncoderDataSize:int = 8192
    AutoEncoderLearningRate:float = 0.0001
    AutoEncoderBatchSize:int = 2**0
    AutoEncoderEpochs:int = 4


