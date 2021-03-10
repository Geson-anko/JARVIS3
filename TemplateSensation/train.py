import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from os.path import isfile
import numpy as np
import torch

from .sensation import Sensation
from .sensation_models import AutoEncoder,DeltaTime
from .config import config

from torch_model_fit import Fit 
from MasterConfig import Config
from MemoryManager import MemoryManager
import multiprocessing as mp

class Train(MemoryManager):
    MemoryFormat = Sensation.MemoryFormat
    LogTitle:str = f'train{MemoryFormat}'

    def __init__(self,device:torch.device,debug_mode:bool=False) -> None:
        super().__init__(log_title=self.LogTitle, debug_mode=debug_mode)
        self.device = torch.device(device)
        self.dtype = Sensation.Training_dtype
        self.fit = Fit(self.LogTitle,debug_mode)

    def activation(self,shutdown:mp.Value,sleep:mp.Value) -> None:

        # ------ Additional Trainings ------
        #
        self.release_system_memory()
        # --- end of Additional Training ---



        # ------ AutoEncoder training ------
        # load data for Training AutoEncoder
        names = os.listdir(Sensation.Data_folder)
        if len(names) ==0:
            self.warn('To train AutoEncoder data does not exist')
            return
        
        times = np.sort([float(i) for i in names])[::-1]
        names = [str(i) for i in times]
        uselen = round(Sensation.AutoEncoderDataSize/Sensation.DataSavingRate)
        uses = names[:uselen]
        deletes = names[uselen:]
        for i in deletes:
            self.remove_file(i)
        
        data = np.concatenate([self.load_python_obj(os.path.join(Sensation.Data_folder,i) for i in uses)])
        data = torch.from_numpy(data)
        self.log(data.shape,debug_only=True)
        model = AutoEncoder()
        model.encoder.load_state_dict(torch.load(Sensation.Encoder_params,map_location=self.device))
        model.decoder.load_state_dict(torch.load(Sensation.Decoder_params,map_location=self.device))

        # AutoEncoder settings
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=Sensation.AutoEncoderLearningRate)
        epochs = Sensation.AutoEncoderEpochs
        batch_size = Sensation.AutoEncoderBatchSize
            # Train
        self.fit.Train(
            shutdown,sleep,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            train_x=data,
            train_y=data
            )   
        torch.save(model.encoder.state_dict(),Sensation.Encoder_params)
        torch.save(model.decoder.state_dict(),Sensation.Decoder_params)
        self.log('trained AutoEncoder')
        del data,model
        self.release_system_memory()
        # --- end of AutoEncoder training ---
        
        # ------ DeltaTime training ------
        if not isfile(Sensation.NewestId_file):
            self.warn('To train DeltaTime data does not exist!')
            return None
        newest_id = self.load_python_obj(Sensation.NewestId_file)
        first_id = self.get_firstId(self.MemoryFormat,return_integer=True)
        ids = np.arange(first_id+1,newest_id+1)[:-Sensation.DeltaTimeDataSize]
        if ids.shape[0] == 0:
            self.warn('Not exist memories.')
            return
        ids,data,times = self.load_memory(ids,return_time=True)
        datalen = data.shape[0]
        zerolen = int(np.floor(datalen*Config.deltaT_zero_per))
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
        ans = torch.from_numpy(ans).type(self.dtype).unsqueeze(1)
        self.log(
            'data1:',data1.shape,
            'data2:',data2.shape,
            'ans:',ans.shape,
            debug_only=True
        )
        # Load DeltaTime
        model = DeltaTime()
        model.load_state_dict(torch.load(Sensation.DeltaTime_params,map_location=self.device))

        # DeltaTime settings
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=Sensation.DeltaTimeLearningRate)
        epochs = Sensation.DeltaTimeEpochs
        batch_size = Sensation.DeltaTimeBatchSize

        # Train
        self.fit.Train(
            shutdown,sleep,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion = criterion,
            device=self.device,
            train_x=[data1,data2],
            train_y=ans,
        )
        torch.save(model.state_dict(),Sensation.DeltaTime_params)
        self.log('Trained DeltaTime')
        del data1,data2,ans,model

        self.release_system_memory()

        self.log('Train process was finished')