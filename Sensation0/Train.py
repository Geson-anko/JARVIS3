from debug_tools import Debug
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

import numpy as np
import cv2
from torchvision import transforms
import torch

from .sensation import Sensation
from .configure import config
from .AutoEncoder import AutoEncoder
from .DeltaTime import DeltaT

from torch_model_fit import Fit 
from MasterConfig import Config as mconf
from MemoryManager import MemoryManager

class Train(MemoryManager):
    memory_format:str =Sensation.memory_format
    log_title:str = f'train{memory_format}'
    
    def __init__(self,device:torch.device,debug_mode:bool=False) -> None:
        super().__init__(log_title=self.log_title, debug_mode=debug_mode)
        self.device = torch.device(device)
        self.dtype = config.training_dtype
        self.debug = Debug(self.log_title,debug_mode)
        self.fit = Fit(self.log_title)

    def activation(self,cmd) -> None:

        # load and preprocess data for Training AutoEncoder
        names = os.listdir(config.data_folder)
        times = np.sort([float(i) for i in names])
        names = [str(i) for i in times]
        uses = names[:config.train_video_use]
        deletes = names[config.train_video_use:]
        for i in deletes:
            self.remove_file(i)

        data = np.concatenate([self.load_python_obj(os.path.join(config.data_folder,i)) for i in uses])
        data = self.preprocess(data)
        self.debug.log(data.shape,debug_only=True)

        # load AutoEncoder 
        model = AutoEncoder()
        model.encoder.load_state_dict(torch.load(config.encoder_params,map_location=self.device))
        model.decoder.load_state_dict(torch.load(config.decoder_params,map_location=self.device))

        # AutoEncoder settings
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=config.AE_lr)
        epochs = config.AE_epochs
        batch_size = config.AE_batch_size

        # Train
        self.fit.Train(
            cmd,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            train_x=data,
            train_y=data
            )        
        model = model
        torch.save(model.encoder.state_dict(),config.encoder_params)
        torch.save(model.decoder.state_dict(),config.decoder_params)
        self.debug.log('trained AutoEncoder')
        del data,model
        self.release_system_memory()

        ## training delta time model
        # loading and preproessing dataset
        newest_id = self.load_python_obj(config.newestId_file)
        first_id = self.get_firstId(self.memory_format)
        ids = np.arange(first_id,newest_id)[:config.time_use]
        ids,data,times = self.load_memory(ids,return_time=True)
        datalen = data.shape[0]
        zerolen = int(np.floor(datalen*config.zero_per))
        zero_idx = np.random.permutation(datalen)[:zerolen]
        zero_data = data[zero_idx]
        zero_ans = np.zeros(zerolen,dtype=times.dtype)
        data_idx = np.random.permutation(datalen)
        data_sh = data[data_idx]
        deltatimes = np.abs(times - times[data_idx])
        data_idx = np.random.permutation((datalen+zerolen))
        data1 = np.concatenate([data,zero_data])[data_idx]
        data2 = np.concatenate([data_sh,zero_data])[data_idx]
        ans = np.concatenate([deltatimes,zero_ans])[data_idx]

        data1 = torch.from_numpy(data1).type(self.dtype)
        data2 = torch.from_numpy(data2).type(self.dtype)
        ans = torch.from_numpy(ans).type(self.dtype)
        self.debug.log(
            'data1:',data1.shape,
            'data2:',data2.shape,
            'ans:',ans.shape,
            debug_only=True
        )

        # load deltaT
        model = DeltaT()
        model.load_state_dict(torch.load(config.deltatime_params,map_location=self.device))

        # deltaT settings
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=config.DT_lr)
        epochs = config.DT_epochs
        batch_size = config.DT_batch_size

        # Train
        self.fit.Train(
            cmd,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion = criterion,
            device=self.device,
            train_x=[data1,data2],
            train_y=ans,
        )
        torch.save(model.state_dict(),config.deltatime_params)
        self.debug.log('Trained DeltaTime')
        del data1,data2,ans,model

        self.release_system_memory()

        self.debug.log('Train process was finished')

        



            
            
    def preprocess(self,data:np.ndarray) -> torch.Tensor:
        data = torch.from_numpy(data).permute(0,3,2,1)
        resizer = transforms.Resize(config.frame_size)
        data = resizer(data).type(self.dtype) / 255
        return data

