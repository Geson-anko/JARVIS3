import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from os.path import join as pathjoin
import torch
import numpy as np

from OutputBase import OutputBase
from Sensation7 import Sensation
from Sensation7.config import config
from .oshaberi_model import OshaberiText,OshaberiMel,OshaberiWave
import pyaudio
import sentencepiece as spm
import time
from Sensation7.torch_KMeans import torch_KMeans
from concurrent.futures import ThreadPoolExecutor
"""
This file is Output Oshaberi 2! Using MFCC2 kaburi1

"""

class Output(OutputBase):
    LogTitle:str = 'OutputOshaberi'
    UsingMemoryFormat:str = '7'
    MaxMemoryLength:int = config.use_mem_len
    UseMemoryLowerLimit:int = config.use_mem_len

    SleepWaitTime:float = 2
    MaxFrameRate:int = 30

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    OshaberiText_params:str = pathjoin(Param_folder,'OshaberiText2021-04-18_08-54-35_mfcc2_cent96.params')
    OshaberiMel_params:str = pathjoin(Param_folder,'OshaberiMel2021-04-17_15-34-52-569810_mfcc2_cent96.params')
    OshaberiWave_params:str = pathjoin(Param_folder,'OshaberiWave2021-04-15_23-56-18-542972.params')

    dtype = np.float16
    torchdtype = torch.float16

    def LoadModels(self) -> None:
        self.OshaberiText = self.LoadPytorchModel(OshaberiText(),self.OshaberiText_params,self.torchdtype)

        self.FTmodel = self.load_python_obj(Sensation.FastText_params)
        self.log('loaded Fasttext.')

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(Sensation.Separator_params)
        self.log('loaded SentencePiece separator.')

        # set initial inputs
        self.init_current = torch.zeros(OshaberiText.input_current,dtype=self.torchdtype,device=self.device)
        bos = self.sp.IdToPiece(self.sp.bos_id())
        self.init_current[0][0] = torch.from_numpy(np.array(self.FTmodel.wv[bos]))


    def Start(self) -> None:
        self.centroids = torch.load(Sensation.Centroids_params,map_location='cpu')
        self.log('loaded Centroids')

        self.OshaberiMel = self.LoadPytorchModel(OshaberiMel(),self.OshaberiMel_params,self.torchdtype)
        self.OshaberiWave = self.LoadPytorchModel(OshaberiWave(),self.OshaberiWave_params,self.torchdtype)

        # get sirence centroid
        kmeas = torch_KMeans(0)
        sirence = torch.zeros_like(self.centroids[:1])
        classes = kmeas.clustering(self.centroids,sirence)[0]
        self.sirence = self.centroids[classes]
        self.log('Got sirence centroids')
        del kmeas,sirence,classes

        # set audio streamings
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=config.pyaudio_format,
            channels=config.channels,
            rate=config.speak_fps,
            output=True,
        )        
        self.log('setted audio streamer')
        # chars reading
        with open(Sensation.Chars_file,'r',encoding='utf-8') as f:
            self.charactors = f.read()[:self.centroids.size(0)]
        self.log('loaded charactors file')

        # Speak values
        self.executor = ThreadPoolExecutor(1)
        self.SpeakVoice = None
        self.speaker_result = self.executor.submit(self.Speak)


    def Update(self, MemoryData: torch.Tensor) -> None:
        memory = MemoryData.unsqueeze(0)
        current = self.init_current.clone()
        text = ''
        words_num = 0
        for _ in range(config.generate_max_words-1):
            out = self.OshaberiText(current,memory)
            word_idx = torch.argmax(out.view(-1)).item()
            if word_idx == self.sp.unk_id():
                continue
            elif word_idx == self.sp.eos_id():
                break
            else:
                word = self.sp.IdToPiece(word_idx)
                text += word
                if word in self.FTmodel.wv:
                    vector = self.FTmodel.wv[word]
                    words_num += 1
                    vector = torch.from_numpy(np.array(vector))
                    current[0][words_num] = vector
                else:
                    break
            time.sleep(config.recognize_second)
        voicevec = []
        text = self.Zenkaku2Hankaku(text)
        print(self.LogTitle,text)
        for i in text:
            if i in self.charactors:
                voicevec.append(self.centroids[self.charactors.index(i)])
        vlen = len(voicevec)
        if vlen == 0:
            return
        padlen = config.speak_seq_len - (vlen%config.speak_seq_len)
        voicevec += [self.sirence]*padlen
        kiritorilen = config.recognize_length * padlen
        voicevec = torch.stack(voicevec).view(-1,config.speak_seq_len,config.MFCC_channels).type(self.torchdtype)
        melvoice = [self.OshaberiMel(self.ToDevice(i).unsqueeze(0)).to('cpu') for i in voicevec]
        wavevoice = torch.cat([self.OshaberiWave(self.ToDevice(i)).to('cpu') for i in melvoice])
        wavevoice = wavevoice.detach().numpy().reshape(-1)[:-kiritorilen]
        wavevoice = np.round(wavevoice*config.sample_range).astype(config.audio_dtype)
        if self.SpeakVoice is None:
            self.SpeakVoice = wavevoice
        else:
            #print('skipped')
            pass

    def UpdateEnd(self) -> None:
        pass

    def End(self) -> None:
        self.actting = False
        self.executor.shutdown(True)
        self.speaker_result.result()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

    hankakukatakana = 'ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜｦﾝ'
    zenkakukatakana = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
    def Zenkaku2Hankaku(self,text:str) -> str:
        chars = []
        for i in text:
            if i in self.zenkakukatakana:
                i = self.hankakukatakana[self.zenkakukatakana.index(i)]
            chars.append(i)
        return ''.join(chars)

    actting:bool = True
    def Speak(self):
        self.log('Speaker is active.')
        while self.actting:
            if self.SpeakVoice is not None:
                voice = self.SpeakVoice.tobytes()
                self.SpeakVoice = None
                self.stream.write(voice)
            else:
                time.sleep(0.01)
        self.log('closed speaker.')


        # Trainig settings
    Training_dtype:torch.dtype = torch.float16
    CorpusUseLength = 2000
    OshaberiTextDataSize:int = 512*100
    OshaberiTextLearningRate:float = 0.001
    OshaberiTextBatchSize:int = 64
    OshaberiTextEpochs:int =32
    MaxSamples = 32
