import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))

from os.path import isfile
import numpy as np
import torch
from .config import config
from Sensation7 import Sensation
from Sensation7.sensation_models import Encoder
from .oshaberi_model import OshaberiText
from .output import Output
from Sensation7.config import config

import sentencepiece as spm
import os
from typing import Tuple

from TrainBase import TrainBase

class Train(TrainBase):
    LogTitle:str = 'TrainOshaberi'
    dtype:torch.dtype = torch.float16
    torchdtype:torch.dtype = torch.float32
    def TrainProcess(self) -> None:
        # ------ Additional Trainings ------
        current,memory,answer= self.GetOshaberiTextData()
        self.log('shape of current,memory,answer',current.shape,memory.shape,answer.shape)
    
        model = self.LoadPytorchModel(OshaberiText(),Output.OshaberiText_params,dtype=torch.float32,eval_mode=False)
        batch_size = Output.OshaberiTextBatchSize
        epochs = Output.OshaberiTextEpochs
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=Output.OshaberiTextLearningRate)
        self.Train(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            train_x=[current,memory],
            train_y=answer,
            metrics=self.fit.CE_Accuracy,
        )
        torch.save(model.state_dict(),Output.OshaberiText_params)
        del model,current,memory,answer
        self.log('trained Text Generator.')
        self.release_system_memory()
        # --- end of Additional Training ---

        self.log('Train process was finished')

    @torch.no_grad()
    def GetOshaberiTextData(self) -> Tuple[torch.Tensor,...]:
        self.FTmodel = self.load_python_obj(Sensation.FastText_params)
        self.separator = spm.SentencePieceProcessor()
        self.separator.Load(Sensation.Separator_params)
        self.bos_id = self.separator.bos_id()
        self.eos_id = self.separator.eos_id()
        self.unk_id = self.separator.unk_id()
        self.bos = self.separator.IdToPiece(self.bos_id)
        self.eos = self.separator.IdToPiece(self.eos_id)

        self.encoder = self.LoadPytorchModel(Encoder(),Sensation.Encoder_params,dtype=self.dtype,device=self.device)
        self.padvec = np.zeros((config.word_dim,),dtype='float16')
        self.log('loaded FastText Sentencepiece,TextEncoder')
        
        with open(Sensation.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.read().split('\n')
        idx = np.random.permutation(len(corpus))[:Output.CorpusUseLength]
        corpus = [corpus[i] for i in idx]
        self.log('Using corpus length is',len(corpus))
        cur,mem,ans = [],[],[]
        for c in corpus:
            i,j,k = self.get_a_OshaberiText_data(c)
            cur.append(i)
            mem.append(j)
            ans.append(k)
        cur = torch.cat(cur)[:Output.OshaberiTextDataSize]
        mem = torch.cat(mem)[:Output.OshaberiTextDataSize]
        ans = torch.cat(ans)[:Output.OshaberiTextDataSize]
        del self.FTmodel,self.separator,corpus,self.encoder,self.padvec
        self.release_system_memory()
        return cur,mem,ans

    def get_a_OshaberiText_data(self,indata:str) -> Tuple[torch.Tensor,...]:
        pieces = [self.bos,*self.separator.EncodeAsPieces(indata),self.eos]
        piecesid,vectors = [],[]
        for i in pieces:
            pid =self.separator.PieceToId(i)
            if i in self.FTmodel.wv and pid != self.separator.unk_id():
                piecesid.append(pid)
                vectors.append(self.FTmodel.wv[i])
        answer = np.array(piecesid[1:],dtype='int64')
        answer = torch.from_numpy(answer)

        current = []
        veclen = len(vectors)
        for i in range(1,veclen):
            vec = vectors[:i]
            if len(vec) < config.generate_max_words:
                vec += [self.padvec]*(config.generate_max_words - len(vec))
            else:
                vec = vec[-config.generate_max_words:]
            current.append(np.stack(vec))
        current = np.stack(current)
        current = torch.from_numpy(current).type(self.dtype)

        memvec = []
        for i in range(veclen):
            vec = vectors[i:i+config.text_seq_len]
            if len(vec) < config.text_seq_len:
                vec += [self.padvec] * (config.text_seq_len- len(vec))
            memvec.append(np.stack(vec))
        memvec = torch.from_numpy(np.stack(memvec)).type(self.dtype)
        memvec = self.fit.Predict(self.encoder,memvec,Output.MaxSamples,self.device)
        mvl = memvec.size(0)
        if mvl < config.use_mem_len:
            memvec = memvec.repeat(((config.use_mem_len//mvl)+1,1))
            mvl = memvec.size(0)
        memory = torch.stack([memvec[np.random.permutation(mvl)[:config.use_mem_len]] for _ in range(current.size(0))])
        sample_idx = np.random.permutation(answer.size(0))[:Output.MaxSamples]
        memory = memory[sample_idx]
        current = current[sample_idx]
        answer = answer[sample_idx]
        return current,memory,answer