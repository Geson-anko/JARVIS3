import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from os.path import isfile
import numpy as np
import torch

from .sensation import Sensation
from .sensation_models import AutoEncoder
from .config import config

from TrainBase import TrainBase

class Train(TrainBase):
    MemoryFormat = Sensation.MemoryFormat
    LogTitle:str = f'train{MemoryFormat}'

    def TrainProcess(self) -> None:
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
        
        data = np.concatenate([self.load_python_obj(os.path.join(Sensation.Data_folder,i)) for i in uses])
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
        self.Train(
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

        self.log('Train process was finished')