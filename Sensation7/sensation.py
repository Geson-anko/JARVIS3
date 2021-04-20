import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
import math
import torch
from torch.nn import Module
from os.path import join as pathjoin

from .config import config
from SensationBase import SensationBase
from .sensation_models import Encoder,HumanCheck
import pyaudio 
import sentencepiece as spm
import time
from .torch_KMeans import torch_KMeans

from typing import Union


class Sensation(SensationBase):
    MemoryFormat:str = '7'# your process memory format (id[0])
    LogTitle:str = f'sensation{MemoryFormat}'
    ReadOutLength:int = 16384 # ReadOutLength
    KeepLength:int = math.floor(ReadOutLength*0.7)  # ReadOutLength * 0.7
    MemoryListLength:int = math.floor(ReadOutLength*0.01)# 1% of ReadOutLength
    MemorySize:int = abs(int(np.prod(Encoder.output_size)))
    SameThreshold:float = 0.001 # The Threshold of memory error.
    DataSaving:bool = False
    DataSize:tuple = Encoder.input_size[1:]
    DataSavingRate:int = 64

    Encoder:Module = Encoder
    SleepWaitTime:float = 0.1

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    Data_folder:str = pathjoin(Current_directory,'data') # /current_dicrectory/data/
    Temp_folder:str = pathjoin(Current_directory,'temp') # /Current_directory/temp/

    Corpus_file:str = pathjoin(f'Sensation{MemoryFormat}','corpus.txt')
    ## defining parameter file
    Encoder_params:str= pathjoin(Param_folder,'TextEncoder17-04-21_21-14-50_mfcc2_cent96.params') # your encoder parameter file name
    Decoder_params:str = pathjoin(Param_folder,'TextDecoder17-04-21_21-14-50_mfcc2_cent96.params') # yout decoder parameter file name
    Centroids_params:str = pathjoin(Param_folder,'centroids_mfcc2_96_datasize32_2021-04-09 13_44_06_k_means++.tensor')
    FastText_params:str = pathjoin(Param_folder,'cent96fasttext64dim_mfcc2.pkl')
    HumanCheck_params:str = pathjoin(Param_folder,'humancheck.params')
    Chars_file:str = pathjoin(Param_folder,'chars.txt')
    Separator_params:str = pathjoin(f'Sensation{MemoryFormat}/params',f'{config.separator_name}.model')
    

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
        super().LoadModels()
        
        # separator
        self.separator = spm.SentencePieceProcessor()
        self.separator.Load(model_file=self.Separator_params)
        self.bos = self.separator.IdToPiece(self.separator.bos_id())
        self.eos = self.separator.IdToPiece(self.separator.eos_id())
        self.log('loaded SentencePiece separator')

        self.FTmodel = self.load_python_obj(self.FastText_params)
        self.log('loaded Fasttext')
    

    def Start(self) -> None:
        # This method is called when process start.
        self.humanchecker = self.ToDevice(HumanCheck().type(self.torchdtype))
        self.humanchecker.load_state_dict(torch.load(self.HumanCheck_params))
        self.log('loaded HumanCheck')

        self.centroids = torch.load(self.Centroids_params,map_location='cpu')
        self.centroids = self.ToDevice(self.centroids)
        self.log('loaded Centroids')

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

        # mel filter bank
        self.mel_filter_bank = self.get_melFilterBank(config.frame_rate,config.recognize_length,config.MFCC_channels).T
        self.mel_filter_bank = self.ToDevice(torch.from_numpy(self.mel_filter_bank)).float()

    textstart:bool = False
    textend:bool = False
    @torch.no_grad()
    def Update(self) -> Union[None,np.ndarray]:
        # This method is called every frame start.
        # your data process
        data = self.stream.read(config.sample_length)
        data = np.frombuffer(data,config.audio_dtype).reshape(-1)
        data = data /config.sample_range
        data = torch.from_numpy(data).type(self.torchdtype).view(1,config.seq_len,config.recognize_length)
        data = self.ToDevice(data)
        isHuman = (self.humanchecker(data).view(-1) > config.HumanThreshold).item()
        vectors = None
        if isHuman:
            self.textstart = True
            data = data.squeeze().float()
            data[:,1:] =  data[:,1:] - config.MFCC_p*data[:,:-1]
            data = torch.abs(torch.fft.rfft(data))
            data = torch.mm(data,self.mel_filter_bank)
            data = torch.log10(data+1)
            classes = self.kmeans.clustering(self.centroids,data).to('cpu').detach().numpy()
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
                
    def UpdateEnd(self) -> None:
        # This method is called every frame end.
        pass

    def End(self) -> None:
        # This method is called when shutting down process.
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    def SleepProcess(self) -> None:
        with open(self.Corpus_file,'r',encoding='utf-8') as f:
            corpus = f.readlines()[-config.use_corpus_length:]
        with open(self.Corpus_file,'w',encoding='utf-8') as f:
            f.writelines(corpus)

        return super().SleepProcess()
    # ------ train settings ------
    Training_dtype:torch.dtype = torch.float16
    AutoEncoderDataSize:int = 8192
    AutoEncoderLearningRate:float = 0.0001
    AutoEncoderBatchSize:int = 2**0
    AutoEncoderEpochs:int = 4

    # Mel Filter bank
    def hz2mel(self,f):
        """Hzをmelに変換"""
        return 2595 * np.log(f / 700 + 1.0)

    def mel2hz(self,m):
        """melをhzに変換"""
        return 700 * (np.exp(m / 2595) - 1.0)

    def get_melFilterBank(self,fs, N, numChannels):
        """メルフィルタバンクを作成"""
        # ナイキスト周波数（Hz）
        fmax = fs / 2
        # ナイキスト周波数（mel）
        melmax = self.hz2mel(fmax)
        # 周波数インデックスの最大数
        nmax = math.floor(N/2) + 1
        # 周波数解像度（周波数インデックス1あたりのHz幅）
        df = fs / N
        # メル尺度における各フィルタの中心周波数を求める
        dmel = melmax / (numChannels + 1)
        melcenters = np.arange(1, numChannels + 1) * dmel
        # 各フィルタの中心周波数をHzに変換
        fcenters = self.mel2hz(melcenters)
        # 各フィルタの中心周波数を周波数インデックスに変換
        indexcenter = np.round(fcenters / df)
        # 各フィルタの開始位置のインデックス
        indexstart = np.hstack(([0], indexcenter[0:numChannels - 1]))
        # 各フィルタの終了位置のインデックス
        indexstop = np.hstack((indexcenter[1:numChannels], [nmax]))
        filterbank = np.zeros((numChannels, nmax))
        for c in range(0, numChannels):
            # 三角フィルタの左の直線の傾きから点を求める
            increment= 1.0 / (indexcenter[c] - indexstart[c])
            for i in range(int(indexstart[c]), int(indexcenter[c])):
                filterbank[c, i] = (i - indexstart[c]) * increment
            # 三角フィルタの右の直線の傾きから点を求める
            decrement = 1.0 / (indexstop[c] - indexcenter[c])
            for i in range(int(indexcenter[c]), int(indexstop[c])):
                filterbank[c, i] = 1.0 - ((i - indexcenter[c]) * decrement)

        return filterbank


