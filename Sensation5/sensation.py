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
from Sensation1.config import config as vconf

import cv2
from torchvision import transforms
from typing import Union,Tuple

class Sensation(SensationBase):
    MemoryFormat:str = '5'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = 32768 # ReadOutLength
    KeepLength:int = math.floor(ReadOutLength*0.7)  # ReadOutLength * 0.7
    MemoryListLength:int = math.floor(ReadOutLength*0.01)# 1% of ReadOutLength
    MemorySize:int = int(np.prod(Encoder.output_size))
    SameThreshold:float = 0.13 # The Threshold of memory error.
    DataSize:tuple = Encoder.input_size[1:]
    DataSavingRate:int = 128
    MaxFrameRate:int = config.frame_rate

    Encoder:Module = Encoder
    SleepWaitTime:float = 3

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    Data_folder:str = pathjoin(Current_directory,'data') # /current_dicrectory/data/
    Temp_folder:str = pathjoin(Current_directory,'temp') # /Current_directory/temp/

    ## defining parameter file
    Encoder_params:str= pathjoin(Param_folder,'Encoder.params') # your encoder parameter file name
    Decoder_params:str = pathjoin(Param_folder,'Decoder.params') # yout decoder parameter file name

    ## defining temporary file
    NewestId_file:str = pathjoin(Temp_folder,'NewestId.pkl')
    MemoryList_file:str = pathjoin(Temp_folder,'MemoryList.pkl')
    ReadOutMemory_file:str = pathjoin(Temp_folder,'ReadOutMemory.pkl')
    ReadOutId_file:str = pathjoin(Temp_folder,'ReadOutId.pkl')
    ReadOutTime_file:str = pathjoin(Temp_folder,'ReadOutTime.pkl')

    dtype:np.dtype = np.float16  # data type
    torchdtype:torch.dtype = torch.float16 # torch tensor data type

    # ------ your settings ------
    

    def Start(self) -> None:
        # This method is called when process start.
        self.capture = cv2.VideoCapture(vconf.default_video_capture,cv2.CAP_DSHOW)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,config.frame_height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,config.frame_width)
        self.capture.set(cv2.CAP_PROP_FPS,config.frame_rate)
        self.resizer = transforms.Resize(config.frame_size)

        if not self.capture.isOpened():
            self.exception('cannot open video capture! Please check default_video_capture.')
        else:
            ret,old_img = self.capture.read()
            old_img = cv2.resize(old_img,config.frame_size)
            self.old_img = torch.from_numpy(old_img).to(self.device) /255

        h,w = np.arange(config.frame_height),np.arange(config.frame_width)
        self.grid_h,self.grid_w = [i.reshape(-1).astype('int') for i in np.meshgrid(h,w)]
        self.grid_idx = np.arange(config.frame_width*config.frame_height)
        tp = np.random.randint(0,config.frame_width*config.frame_height)
        self.old_point = (self.grid_h[tp],self.grid_w[tp]) 

    def Update(self) -> torch.Tensor:
        # This method is called every frame start.
        # your data process
        ret,img = self.capture.read()
        img = cv2.resize(img,config.frame_size)
        img_t = torch.from_numpy(img).to(self.device)/255
        point = self.get_point(self.old_img,img_t)
        point = self.next_point(point,self.old_point)
        self.old_img = img_t.clone()
        self.old_point = point
        img,window_img = self.write_get_window(img,img_t,point)
        cv2.imshow(self.LogTitle,img)
        window_img = window_img.permute(2,1,0).unsqueeze(0)

        return window_img

    def UpdateEnd(self) -> None:
        # This method is called every frame end.
        #print('\rcurrent_length',self.current_length,'mins',self.mins,end='')
        cv2.waitKey(1)


    def End(self) -> None:
        # This method is called when shutting down process.
        self.capture.release()
        cv2.destroyAllWindows()


    def get_point(self,img1:torch.Tensor,img2:torch.Tensor) -> tuple:
        submap = torch.mean((img1-img2)**2,dim=-1).view(-1)
        try:
            tp = torch.multinomial(submap,1).to('cpu').numpy()[0]
        except RuntimeError:
            tp = np.random.randint(0,submap.size(0))
        return self.grid_h[tp],self.grid_w[tp]

    def next_point(self,new_point:tuple,old_point:tuple) -> tuple:
        n_x,n_y = new_point
        o_x,o_y = old_point
        x = self.split1_n(o_x,n_x,config.kyorokyoro)
        y = self.split1_n(o_y,n_y,config.kyorokyoro)
        return x,y

    def split1_n(self,a,b,n):
        p = (n*a+b)/(1+n)
        return np.round(p).astype(int)
    
    def write_get_window(self,img:np.ndarray,torch_img:torch.Tensor,point:Tuple[int,int]) -> Tuple[np.ndarray,torch.Tensor]:
        h,w = point
        h -= config.win_point_height
        w -= config.win_point_width
        fh,fw = img.shape[:2]
        fh -= config.window_height
        fw -= config.window_width
        if fh < 0 or fw < 0:
            self.exception(f'Input image is to small! input image size is {(fh,fw)},window size is {(config.window_height,config.window_width)}')
        if h < 0:
            h=0
        elif h > fh:
            h= fh
        if w < 0:
            w = 0
        elif w > fw:
            w = fw
        img[h:h+config.window_height,w:w+config.line_width] = (255 - img[h:h+config.window_height,w:w+config.line_width].copy())
        img[h:h+config.window_height,(w+config.window_width-config.line_width):(w+config.window_width)] = (255 - img[h:h+config.window_height,(w+config.window_width-config.line_width):(w+config.window_width)].copy())
        img[h:h+config.line_width,w:w+config.window_width] = (255-img[h:h+config.line_width,w:w+config.window_width].copy())
        img[(h+config.window_height-config.line_width):(h+config.window_height),w:w+config.window_width] = (255 - img[(h+config.window_height-config.line_width):(h+config.window_height),w:w+config.window_width].copy())

        window = torch_img[h:h+config.window_height,w:w+config.window_width]
        return img,window

    # ------ train settings ------
    Training_dtype:torch.dtype = torch.float16
    AutoEncoderDataSize:int = 2**16
    AutoEncoderLearningRate:float = 0.0001
    AutoEncoderBatchSize:int = 4096
    AutoEncoderEpochs:int = 20


