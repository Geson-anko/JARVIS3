from MemoryManager import MemoryManager
from MasterConfig import Config as mconf
from Sensation6 import Sensation
from Sensation6.sensation_models import KikitoriEncoder,Encoder
from Sensation6.torch_KMeans import torch_KMeans
from Sensation6.config import config
from .output import Output 
from .output_models import Oshaberi,TextGenerator
from torch_model_fit import Fit

import numpy as np
import multiprocessing as mp
import torch
import sentencepiece as spm
import os
from pydub import AudioSegment
from typing import Tuple,Union

class Train(MemoryManager):
    MemoryFormat = Sensation.MemoryFormat
    LogTitle:str = f'TrainOshaberi'

    def __init__(self,device:torch.device,debug_mode:bool=False) -> None:
        super().__init__(log_title=self.LogTitle, debug_mode=debug_mode)
        self.device = torch.device(device)
        self.dtype = Sensation.Training_dtype
        self.fit = Fit(self.LogTitle,debug_mode)

    def activation(self,shutdown:mp.Value,sleep:mp.Value) -> None:
        
        # ----- Oshaberi Training ------
        data,ans = self.GetOshaberiData()
        self.log('shape of data shape',data.shape,ans.shape)
        
        model = Oshaberi()
        model.load_state_dict(torch.load(Output.Oshaberi_file,map_location=self.device))
        batch_size = Output.OshaberiBatchSize
        epochs = Output.OshaberiEpochs
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=Output.OshaberiLearningRate)
        self.fit.Train(
            shutdown,sleep,
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            train_x=data,
            train_y=ans,
        )
        torch.save(model.state_dict(),Output.Oshaberi_file)
        del model,data,ans
        self.release_system_memory()
        self.log('trained Oshaberi.')



        # --- end of Oshaberi Training ---
        if shutdown.value or not sleep.value:
            self.log('Train process was stopped')
            return
        # ------ TextGenerator Training -------
        current,memory,answer= self.GetTextGeneratorData()
        self.log('shape of current,memory,answer',current.shape,memory.shape,answer.shape)
        
        model = TextGenerator()
        model.load_state_dict(torch.load(Output.TextGenerator_file,map_location=self.device))
        batch_size =Output.TextGeneratorBatchSize
        epochs = Output.TextGeneratorEpochs
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(),lr=Output.TextGeneratorLearningRate)

        self.fit.Train(
            shutdown,sleep,
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
        torch.save(model.state_dict(),Output.TextGenerator_file)
        del model,current,memory,answer
        self.log('trained Text Generator.')
        self.release_system_memory()
        # --- end of TextGenerator Training ---

        if shutdown.value or not sleep.value:
            self.log('Train process was stopped')
            return

    @torch.no_grad()
    def GetOshaberiData(self) -> Tuple[torch.Tensor,...]:
        self.kencoder = KikitoriEncoder().to(self.device).type(torch.float32)
        self.kencoder.load_state_dict(torch.load(Sensation.KikitoriEncoder_params,map_location=self.device))
        self.kencoder.eval()

        self.centroids = torch.load(Sensation.Centroids_file,map_location=self.device).to(self.device)
        self.kmeans = torch_KMeans(0)
        self.inlen = int(config.speak_second*config.frame_rate)
        self.outlen = int(config.speak_seq_len*self.centroids.size(-1))
        self.batch_size = 512
        self.log('loaded KikitoriEncoder,Centroids,')


        files = [os.path.join(Output.Voice_folder,i) for i in os.listdir(Output.Voice_folder)]
        self.log('Using file number is',len(files))

        data,ans = [],[]
        for f in files[:1]:
            i,j = self.get_a_oshaberi_data(f)
            data.append(i)
            ans.append(j)
        data = torch.cat(data)[:Output.OshaberiDataSize]
        ans = torch.cat(ans)[:Output.OshaberiDataSize]
        del self.kencoder,self.centroids
        self.release_system_memory()
        return data,ans

    
    def get_a_oshaberi_data(self,indata) -> Tuple[torch.Tensor,...]:
        sound = AudioSegment.from_file(indata)
        if sound.channels != config.channels:
            sound = sound.set_channels(config.channels)
        if sound.sample_width != config.sample_width:
            sound = sound.set_sample_width(config.sample_width)

        insound = np.array(sound.set_frame_rate(config.frame_rate).get_array_of_samples())
        padlen = self.inlen - (insound.shape[0] % self.inlen)
        pad = np.zeros(padlen,dtype=insound.dtype)
        insound = np.concatenate([insound,pad]).reshape(-1,config.recognize_length)
        insound = (insound / config.sample_range).astype('float32')
        insound = torch.from_numpy(insound).unsqueeze(1)
        encoded = Fit.Predict(self.kencoder,insound,self.batch_size,self.device)
        encoded = encoded.type(torch.float32).to(self.device)
        classes = self.kmeans.predict(self.centroids,encoded,batch_size=self.batch_size,device=self.device)
        data = self.centroids.to('cpu')[classes].type(torch.float16).view(-1,1,self.outlen)

        outsound = np.array(sound.set_frame_rate(config.speak_fps).get_array_of_samples())
        outsound = (outsound/config.speak_range).astype('float16')
        padlen = config.speak_length - (outsound.shape[0] % config.speak_length)
        pad = np.zeros(padlen,dtype=outsound.dtype)
        ans= np.concatenate([outsound,pad]).reshape(-1,1,config.speak_length)
        ans = torch.from_numpy(ans).type(self.dtype)
        return data,ans

    @torch.no_grad()
    def GetTextGeneratorData(self) -> Tuple[torch.Tensor,...]:
        self.FTmodel = self.load_python_obj(Sensation.FastText_file)
        self.separator = spm.SentencePieceProcessor()
        self.separator.Load(Sensation.Separator_file)
        self.bos_id = self.separator.bos_id()
        self.eos_id = self.separator.eos_id()
        self.unk_id = self.separator.unk_id()
        self.bos = self.separator.IdToPiece(self.bos_id)
        self.eos = self.separator.IdToPiece(self.eos_id)


        self.encoder = Encoder().to(self.device).type(self.dtype)
        self.encoder.load_state_dict(torch.load(Sensation.Encoder_params,map_location=self.device))
        self.encoder.eval()
        self.padvec = np.zeros((config.word_dim,),dtype='float16')

        self.log('loaded FastText Sentencepiece,TextEncoder')

        with open(Sensation.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.read().split('\n')
        idx = np.random.permutation(len(corpus))[:Output.CorpusUseLength]
        corpus = [corpus[i] for i in idx]
        self.log('Using corpus length is',len(corpus))
        cur,mem,ans = [],[],[]
        for c in corpus:
            i,j,k = self.get_a_textgenerator_data(c)
            cur.append(i)
            mem.append(j)
            ans.append(k)
        cur = torch.cat(cur)[:Output.TextGeneratorDataSize]
        mem = torch.cat(mem)[:Output.TextGeneratorDataSize]
        ans = torch.cat(ans)[:Output.TextGeneratorDataSize]
        del self.FTmodel,self.separator,corpus,self.encoder,self.padvec
        self.release_system_memory()
        return cur,mem,ans

    def get_a_textgenerator_data(self,indata:str) -> Tuple[torch.Tensor,...]:
        pieces = [self.bos,*self.separator.EncodeAsPieces(indata),self.eos]
        piecesid,vectors = [],[]
        for i in pieces:
            pid =self.separator.PieceToId(i)
            if i in self.FTmodel.wv and pid != self.separator.unk_id():
                piecesid.append(pid)
                vectors.append(self.FTmodel.wv[i])
        answer = np.array(piecesid[1:],dtype='int64')
        answer = torch.from_numpy(answer)


        current= []
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
        memvec = Fit.Predict(self.encoder,memvec,Output.MaxSamples,self.device)
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

