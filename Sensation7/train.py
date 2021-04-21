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
import sentencepiece as spm

class Train(TrainBase):
    MemoryFormat = Sensation.MemoryFormat
    LogTitle:str = f'train{MemoryFormat}'

    def TrainProcess(self) -> None:
        # ------ Separator training ------
        spm.SentencePieceTrainer.Train(f"--input={Sensation.Corpus_file} --model_prefix=Sensation{Sensation.MemoryFormat}/params/{config.separator_name} --vocab_size={config.vocab_size}")
        # --- end of SentencePiece Trainings ---
        self.release_system_memory()

        # ------ Fasttext training ------
        separator = spm.SentencePieceProcessor()
        separator.Load(f'Sensation{Sensation.MemoryFormat}/params/{config.separator_name}.model')
        bos = separator.IdToPiece(separator.bos_id())
        eos = separator.IdToPiece(separator.eos_id())

        with open(Sensation.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.read().split('\n')[-Sensation.CorpusUseLength:]
        words = [[bos,*separator.EncodeAsPieces(i),eos] for i in corpus]
        FTmodel = self.load_python_obj(Sensation.FastText_params)
        FTmodel.build_vocab(words,update=True)
        FTmodel.train(sentences=words,total_examples=len(words),epochs=Sensation.FasttextEpochs)
        self.save_python_obj(Sensation.FastText_params,FTmodel)
        self.log('trained Fasttext')
        # --- end of Fasttext training ---

        # ------ AutoEncoder training ------
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
        idx = np.random.permutation(len(data))[:Sensation.AutoEncoderDataSize]
        data = data[idx]

        self.log('Text AutoEncoder data shape',data.shape)
        del FTmodel,separator
        self.release_system_memory()
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
        # ----- corpus reducing ------
        with open(Sensation.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.readlines()[-Sensation.SavingCorpusLength:]
        with open(Sensation.Corpus_file,'w',encoding='utf-8') as f:
            f.writelines(corpus)
        self.log('reduced corpus')
        self.log('Train process was finished')