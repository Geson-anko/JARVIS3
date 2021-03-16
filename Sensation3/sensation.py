import os
import sys

from torch._C import dtype
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import math
import torch
from torch.nn import Module
from os.path import join as pathjoin

from .config import config
from SensationBase import SensationBase
from .sensation_models import Encoder,KikitoriEncoder,HumanCheck
from .torch_KMeans import torch_KMeans

import pyaudio
import sentencepiece as spm
from gensim.models import FastText
import time
from concurrent.futures import ThreadPoolExecutor

class Sensation(SensationBase):
    MemoryFormat:str = '3'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = 16384 # ReadOutLength
    KeepLength:int = math.floor(ReadOutLength*0.7)  # ReadOutLength * 0.7
    MemoryListLength:int = math.floor(ReadOutLength*0.01)# 1% of ReadOutLength
    MemorySize:int = int(np.prod(Encoder.output_size))
    SameThreshold:float = 0.0 # The Threshold of memory error.
    DataSize:tuple = Encoder.input_size[1:]
    DataSavingRate:int = 2

    Encoder:Module = Encoder
    SleepWaitTime:float = 0.1

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    Data_folder:str = pathjoin(Current_directory,'data') # /current_dicrectory/data/
    Temp_folder:str = pathjoin(Current_directory,'temp') # /Current_directory/temp/

    KikitoriData_folder:str = pathjoin(Current_directory,'kikitori')
    Corpus_file:str = pathjoin('Sensation3','corpus.txt')

    ## defining parameter file
    Encoder_params:str= pathjoin(Param_folder,'EncoderText.params') # your encoder parameter file name
    Decoder_params:str = pathjoin(Param_folder,'DecoderText.params') # yout decoder parameter file name
    KikitoriEncoder_params:str = pathjoin(Param_folder,'EncoderKikitori.params')
    KikitoriDecoder_params:str = pathjoin(Param_folder,'DecoderKikitori.params')
    Centroids_file:str = pathjoin(Param_folder,'centroids_96_datasize320.tensor')
    FastText_file:str = pathjoin(Param_folder,'FastText.pkl')
    Separator_file:str = pathjoin('Sensation3/params',f'{config.separator_name}.model')
    HumanCheck_file:str = pathjoin(Param_folder,'humancheck.params')
    Chars_file:str = pathjoin(Param_folder,'chars.txt')

    ## defining temporary file
    NewestId_file:str = pathjoin(Temp_folder,'NewestId.pkl')
    MemoryList_file:str = pathjoin(Temp_folder,'MemoryList.pkl')
    ReadOutMemory_file:str = pathjoin(Temp_folder,'ReadOutMemory.pkl')
    ReadOutId_file:str = pathjoin(Temp_folder,'ReadOutId.pkl')
    ReadOutTime_file:str = pathjoin(Temp_folder,'ReadOutTime.pkl')

    dtype:np.dtype = np.float16  # data type
    torchdtype:torch.dtype = torch.float16 # torch tensor data type

    # ------ your settings ------
    
    def LoadModels(self) -> None:
        self.encoder = Encoder().to(self.device).type(self.torchdtype)
        self.encoder.load_state_dict(torch.load(self.Encoder_params,map_location=self.device))
        self.log('loaded Encoder')
        
        self.kikitoriencoder = KikitoriEncoder().to(self.device).type(self.torchdtype)
        self.kikitoriencoder.load_state_dict(torch.load(self.KikitoriEncoder_params,map_location=self.device))
        self.log('loaded KikitoriEncoder')

        self.centroids = torch.load(self.Centroids_file,map_location=self.device).to(self.device)
        self.log('loaded Centroids')

        self.separator = spm.SentencePieceProcessor()
        self.separator.Load(model_file=self.Separator_file)
        self.bos = self.separator.IdToPiece(self.separator.bos_id())
        self.eos = self.separator.IdToPiece(self.separator.eos_id())
        self.log('loaded SentencePiece separator')

        self.FTmodel = self.load_python_obj(self.FastText_file)
        self.log('loaded Fasttext')

        return 

    def Start(self) -> None:
        # This method is called when process start.
        self.humanchecker = HumanCheck().to(self.device).type(self.torchdtype)
        self.humanchecker.load_state_dict(torch.load(self.HumanCheck_file))
        self.log('loaded HumanCheck')

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=config.pyaudio_format,
            channels=config.channels,
            rate=config.frame_rate,
            frames_per_buffer=config.sample_length,
            input=True,
        )
        self.kmeans = torch_KMeans(0)
        with open(self.Chars_file,'r',encoding='utf-8') as f:
            self.charactors = f.read()
        self.text = ''

        self.executer = ThreadPoolExecutor()
        if not os.path.isdir(self.KikitoriData_folder):
            os.makedirs(self.KikitoriData_folder)
            self.log('made Kiktori Data Folder')

    textstart:bool = False
    textend:bool = False
    @torch.no_grad()
    def Update(self) -> torch.Tensor:
        # This method is called every frame start.
        # your data process
        data = self.stream.read(config.sample_length)
        data = np.frombuffer(data,config.audio_dtype).reshape(-1)
        data = data /config.sample_range
        data = torch.from_numpy(data).type(self.torchdtype).to(self.device).view(1,config.seq_len,config.recognize_length)
        isHuman = self.humanchecker(data).view(-1).to('cpu').numpy()[0] > config.HumanThreshold

        vectors = None
        if isHuman:
            self.textstart = True
            data = data.view(KikitoriEncoder.insize)
            name = pathjoin(self.KikitoriData_folder,str(time.time())) # Data Saving kikitori
            self.executer.submit(self.save_python_obj,name,data.to('cpu').numpy()) # Data Saving kikitori
            
            encoded = self.kikitoriencoder(data).view(data.size(0),-1)
            classes = self.kmeans.clustering(self.centroids,encoded).to('cpu').detach().numpy()
            self.text += ''.join([self.charactors[i] for i in classes])
        else:
            if self.textstart:
                self.textend = True
                self.textstart = False
                pieces = [self.bos,*self.separator.EncodeAsPieces(self.text),self.eos]
                vectors = [self.FTmodel.wv[i] for i in pieces if i in self.FTmodel.wv]
                vectors = np.stack(vectors).astype(self.dtype)
                vectors = torch.from_numpy(vectors)
                with open(self.Corpus_file,'a',encoding='utf-8') as f:
                    f.write(f'{self.text}\n')
                #self.log('text',self.text)
            self.text = ''

        return vectors


    @torch.no_grad()
    def MemoryProcess(self,Data) -> None:
        if Data is not None:
            datalen = Data.size(0)
            for i in range(0,datalen,config.text_seq_len):
                self.SavedDataLen = 0
                words = Data[i:i+config.text_seq_len]
                if words.size(0) < config.text_seq_len:
                    pad = torch.zeros((config.text_seq_len - words.size(0),words.size(1)),dtype=words.dtype,device=words.device)
                    words = torch.cat([words,pad])
                super().MemoryProcess(words)
                #self.log('current_length',self.current_length,debug_only=True)
            self.textend = False
        else:
            pass
                
    def End(self):
        self.executer.shutdown(True)
    def DataSavingCheck(self) -> None:pass

    def SleepProcess(self) -> None:
        with open(self.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.readlines()[-config.use_corpus_length:]
        with open(self.Corpus_file,'w',encoding='utf-8') as f:
            f.writelines(corpus)

        return super().SleepProcess()
        
    # ------ train settings ------
    Training_dtype:torch.dtype = torch.float16
    KikitoriAEDataSize:int = 8192
    KikitoriAELearningRate:float = 0.0001
    KikitoriAEBatchSize:int = 1024
    KikitoriAEEpochs:int = 4

    CentroidsMaxIter:int = 100

    CorpusUseLength = 10000
    FasttextEpochs = 10
    AutoEncoderLearningRate:float = 0.0001
    AutoEncoderBatchSize:int = 4096
    AutoEncoderEpochs:int = 10


