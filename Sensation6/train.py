import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from os.path import isfile
import numpy as np
import torch

from .sensation import Sensation
from .sensation_models import AutoEncoder,KikitoriAutoEncoder
from .config import config
from .torch_KMeans import torch_KMeans

from torch_model_fit import Fit 
from MemoryManager import MemoryManager
import multiprocessing as mp
import sentencepiece as spm
from gensim.models import FastText

class Train(MemoryManager):
    MemoryFormat = Sensation.MemoryFormat
    LogTitle:str = f'train{MemoryFormat}'

    def __init__(self,device:torch.device,debug_mode:bool=False) -> None:
        super().__init__(log_title=self.LogTitle, debug_mode=debug_mode)
        self.device = torch.device(device)
        self.dtype = Sensation.Training_dtype
        self.fit = Fit(self.LogTitle,debug_mode)

    def activation(self,shutdown:mp.Value,sleep:mp.Value) -> None:
        
        # ------ kikitoriAutoEncoder Trainings ------
        names = os.listdir(Sensation.KikitoriData_folder)
        if len(names) == 0:
            self.warn('To train KikitoriAutoEncoder data does not exist')
            return
        times = np.sort([float(i) for i in names])[::-1]
        names = [str(i) for i in times]
        uselen = round(Sensation.KikitoriAEDataSize/(config.seq_len*Sensation.KikitoriSavingRate))
        uses = names[:uselen]
        deletes = names[uselen:]
        for i in deletes:
            self.remove_file(os.path.join(Sensation.KikitoriData_folder,i))
        """
        data = np.concatenate([self.load_python_obj(os.path.join(Sensation.KikitoriData_folder,i)) for i in uses])
        data = torch.from_numpy(data).type(Sensation.Training_dtype)
        idx = torch.randperm(data.size(0))
        data = data[idx]
        self.log('Kikitori AutoEncoder data shape',data.shape)

        model = KikitoriAutoEncoder()
        model.encoder.load_state_dict(torch.load(Sensation.KikitoriEncoder_params,map_location=self.device))
        model.decoder.load_state_dict(torch.load(Sensation.KikitoriDecoder_params,map_location=self.device))
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=Sensation.KikitoriAELearningRate)
        epochs=Sensation.KikitoriAEEpochs
        batch_size = Sensation.KikitoriAEBatchSize
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
        torch.save(model.encoder.state_dict(),Sensation.KikitoriEncoder_params)
        torch.save(model.decoder.state_dict(),Sensation.KikitoriDecoder_params)
        self.log('trained Kikitori AutoEncoder')
        self.release_system_memory()

        # --- end of Kikitori AutoEncoder Training ---
        # ------ Centroids Trainings ------
        kmeans = torch_KMeans(0)
        centroids = torch.load(Sensation.Centroids_file).to(self.device)
        data = self.fit.Predict(model.encoder,data,batch_size=batch_size,device=self.device).to(self.device).view(-1,centroids.size(1))
        centroids = kmeans.KMeans(centroids.size(0),data,Sensation.CentroidsMaxIter,default_centroids=centroids)
        torch.save(centroids.to('cpu'),Sensation.Centroids_file)
        self.log('trained Centroids')
        del centroids,model,data
        self.release_system_memory()

        # --- end of Centroids Training ---
        """
        if shutdown.value or not sleep.value:
            self.log('Train process was stopped')
            return

        # ------ Separator training ------
        spm.SentencePieceTrainer.Train(f"--input={Sensation.Corpus_file} --model_prefix=Sensation{Sensation.MemoryFormat}/params/{config.separator_name} --vocab_size={config.vocab_size}")
        # --- end of SentencePiece Trainings ---
        if shutdown.value or not sleep.value:
            self.log('Train process was stopped')
            return
        # ------ Fasttext training ------
        separator = spm.SentencePieceProcessor()
        separator.Load(f'Sensation{Sensation.MemoryFormat}/params/{config.separator_name}.model')
        bos = separator.IdToPiece(separator.bos_id())
        eos = separator.IdToPiece(separator.eos_id())

        with open(Sensation.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.read().split('\n')[-Sensation.CorpusUseLength:]
        words = [[bos,*separator.EncodeAsPieces(i),eos] for i in corpus]
        FTmodel = self.load_python_obj(Sensation.FastText_file)
        FTmodel.build_vocab(words,update=True)
        FTmodel.train(sentences=words,total_examples=len(words),epochs=Sensation.FasttextEpochs)
        self.save_python_obj(Sensation.FastText_file,FTmodel)
        self.log('trained Fasttext')
        # --- end of Fasttext training ---
        if shutdown.value or not sleep.value:
            self.log('Train process was stopped')
            return
        # ------ Training Text AutoEncoder ------
        data = []
        for i in words:
            vector = torch.from_numpy(np.stack([FTmodel.wv[q] for q in i if q in FTmodel.wv])).type(Sensation.Training_dtype)
            length = vector.size(0)
            for idx in range(0,length,config.text_seq_len):
                d = vector[idx:idx+config.text_seq_len]
                if d.size(0) < config.text_seq_len:
                    pad = torch.zeros((config.text_seq_len - d.size(0)),d.size(1),dtype=d.dtype,device=d.device)
                    d = torch.cat([d,pad])
                data.append(d)
        data = torch.stack(data)

        self.log('Text AutoEncoder data shape',data.shape)
        del FTmodel,separator
        self.release_system_memory()

        model = AutoEncoder()
        model.encoder.load_state_dict(torch.load(Sensation.Encoder_params,map_location=self.device))
        model.decoder.load_state_dict(torch.load(Sensation.Decoder_params,map_location=self.device))

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=Sensation.AutoEncoderLearningRate)
        epochs=Sensation.AutoEncoderEpochs
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
        self.log('trained Text AutoEncoder')
        self.release_system_memory()
        # --- end of Text AutoEncoder training ----
        # ----- corpus reducing ------
        with open(Sensation.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.readlines()[-Sensation.SavingCorpusLength:]
        with open(Sensation.Corpus_file,'w',encoding='utf-8') as f:
            f.writelines(corpus)
        self.log('reduced corpus')



        self.log('Train process was finished')