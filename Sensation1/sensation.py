from Sensation0.configure import config
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import torch
from torch.nn import Module
from os.path import join as pathjoin
from torchvision import transforms
import cv2

from SensationBase import SensationBase
from .sensation_models import Encoder,DeltaTime

class Sensation(SensationBase):
    MemoryFormat:str = '1'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = 10000 # ReadOutLength
    KeepLength:int = int(ReadOutLength*0.7) # 70% of ReadOutLength
    MemoryListLength:int = int(ReadOutLength*0.01) # 1% of ReadOutLength
    MemorySize:int = int(np.prod(Encoder.output_size))
    DataSize:tuple = Encoder.input_size[1:]
    DataSavingRate:int = 64

    Encoder:Module = Encoder
    DeltaTime:Module = DeltaTime
    SleepWaitTime:float = 0.3

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    Data_folder:str = pathjoin(Current_directory,'data') # /Current_dicrectory/data/
    Temp_folder:str = pathjoin(Current_directory,'temp') # /Current_directory/temp/

    ## defining parameter file
    Encoder_params:str= pathjoin(Param_folder,'encoder.params') # your encoder parameter file name
    Decoder_params:str = pathjoin(Param_folder,'decoder.params') # yout decoder parameter file name
    DeltaTime_params:str = pathjoin(Param_folder,'deltatime.params') # your deltatime parameter file name

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
        self.capture = cv2.VideoCapture(config.default_video_capture,cv2.CAP_DSHOW)
        if not self.capture.isOpened():
            self.exception('cannot open video capture! Please check default_video_capture.')
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT,config.height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,config.width)
        self.capture.set(cv2.CAP_PROP_FPS,config.frame_rate)
        self.resizer = transforms.Resize(config.frame_size)

    def Update(self) -> torch.Tensor:
        # your data process
        ret,img = self.capture.read()
        if not ret:
            self.exception('can not read image from capture!')
        cv2.imshow(self.LogTitle,img)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device).permute(0,3,2,1)
        img = self.resizer(img).type(self.torchdtype)/255
        return img

    def UpdateEnd(self) -> None:
        cv2.waitKey(1)

    def End(self) -> None:
        self.capture.release()
        cv2.destroyAllWindows()


    # ------ train settings ------
    Training_dtype:torch.dtype = torch.float16
    AutoEncoderDataSize:int = 8192
    AutoEncoderLearningRate:float = 0.0001
    AutoEncoderBatchSize:int = 32
    AutoEncoderEpochs:int = 5
    
    DeltaTimeDataSize:int = 8192
    DeltaTimeLearningRate:float = 0.0001
    DeltaTimeBatchSize:int = 4096
    DeltaTimeEpochs:int = 10
    

