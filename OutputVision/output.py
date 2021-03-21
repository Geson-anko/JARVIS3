import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import torch
import numpy as np

from OutputBase import OutputBase
from .config import config
from Sensation1 import Sensation as vsense,sensation_models as models

import cv2


class Output(OutputBase):
    LogTitle:str = 'VisionOutput'
    UsingMemoryFormat:str = '1'
    MaxMemoryLength:int = 1 

    SleepWaitTime:float = 0.1
    MaxFrameRate:int = 10

    dtype = np.float16
    torchdtype = torch.float16

    def LoadModels(self) -> None:
        self.decoder = models.Decoder().to(self.device).type(self.torchdtype)
        self.decoder.load_state_dict(torch.load(vsense.Decoder_params,map_location=self.device))

    def Start(self) -> None:
        pass

    def Update(self, MemoryData: torch.Tensor) -> None:
        """
        Docs
        """
        out = self.decoder(MemoryData).permute(0,3,2,1).to('cpu').squeeze(0).numpy() * 255
        out = np.round(out).astype('uint8')
        cv2.imshow(self.LogTitle,out)
        cv2.waitKey(1)
        
    def UpdateEnd(self) -> None:
        pass
    def End(self) -> None:
        cv2.destroyAllWindows()
