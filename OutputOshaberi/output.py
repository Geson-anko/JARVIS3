import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import torch
import numpy as np
from os.path import join as pathjoin

from OutputBase import OutputBase
from Sensation6.config import config
from .output_models import Oshaberi,TextGenerator
from Sensation6 import Sensation
from Sensation6.torch_KMeans import torch_KMeans
from Sensation6.sensation_models import KikitoriEncoder
import pyaudio
import sentencepiece as spm
import time

class Output(OutputBase):
    LogTitle:str = 'OutputOshaberi'
    UsingMemoryFormat:str = '6'
    MaxMemoryLength:int = config.use_mem_len
    UseMemoryLowerLimit:int = config.use_mem_len

    SleepWaitTime:float = 1
    MaxFrameRate:int = 30

    Current_directory:str = os.path.dirname(os.path.abspath(__file__)) # /Current_directory/...  from root
    Param_folder:str = pathjoin(Current_directory,'params') # /Current_directory/params/
    Oshaberi_file:str = pathjoin(Param_folder,'oshaberi.params')
    TextGenerator_file:str = pathjoin(Param_folder,'TextGenerator.params')
    Voice_folder:str = pathjoin(Current_directory,'Voices')


    dtype = np.float16
    torchdtype = torch.float16

    def LoadModels(self) -> None:
        self.Oshaberi = Oshaberi().to(self.device).type(self.torchdtype)
        self.Oshaberi.load_state_dict(torch.load(self.Oshaberi_file,map_location=self.device))
        self.Oshaberi.eval()
        self.log('loaded Oshaberi.')

        self.TextGenerator = TextGenerator().to(self.device).type(self.torchdtype)
        self.TextGenerator.load_state_dict(torch.load(self.TextGenerator_file,map_location=self.device))
        self.TextGenerator.eval()
        self.log('loaded TextGenerator.')

        self.FTmodel = self.load_python_obj(Sensation.FastText_file)
        self.log('loaded Fasttext.')

        self.centroids = torch.load(Sensation.Centroids_file,map_location='cpu')
        self.log('loaded Centroids')

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(Sensation.Separator_file)
        self.log('loaded SentencePiece separator.')


        Kencoder = KikitoriEncoder().type(torch.float32)
        Kencoder.load_state_dict(torch.load(Sensation.KikitoriEncoder_params,map_location='cpu'))
        kmeans = torch_KMeans(0)
        sirence = torch.zeros(KikitoriEncoder.input_size)
        sirence = Kencoder(sirence).view(1,-1)
        classes = kmeans.clustering(self.centroids,sirence)[0]
        self.sirence = self.centroids[classes]
        self.log('Got sirence centroids.')
        del Kencoder,kmeans,sirence,classes


    def Start(self) -> None:
        self.init_current = torch.zeros(TextGenerator.input_current,dtype=self.torchdtype,device=self.device)

        bos = self.sp.IdToPiece(self.sp.bos_id())
        self.init_current[0][0] = torch.from_numpy(np.array(self.FTmodel.wv[bos]))
        
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=config.pyaudio_format,
            channels=config.channels,
            rate=config.speak_fps,
            output=True,
        )
        with open(Sensation.Chars_file,'r',encoding='utf-8') as f:
            self.charactors = f.read()


    def Update(self, MemoryData: torch.Tensor) -> None:
        memory = MemoryData.unsqueeze(0)
        current = self.init_current.clone()
        text = ''
        words_num =0
        for _ in range(config.generate_max_words-1):
            out = self.TextGenerator(current,memory)
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
            time.sleep(0.01)
        voicevec = []
        #print(text)
        for i in text:
            if i in self.charactors:
                voicevec.append(self.centroids[self.charactors.index(i)])
        vlen = len(voicevec)
        if vlen == 0:
            return
        padlen = config.speak_seq_len - (vlen%config.speak_seq_len)
        voicevec += [self.sirence] * padlen
        voicevec = torch.stack(voicevec).view(-1,config.speak_length).type(self.torchdtype)
        sound = torch.cat([self.Oshaberi(i.to(self.device).unsqueeze(0)).to('cpu').squeeze(0) for i in voicevec])
        sound = sound.detach().numpy().reshape(-1)
        sound = np.round(sound*config.sample_range).astype(config.audio_dtype)
        self.stream.write(sound.tobytes())

    def UpdateEnd(self) -> None:
        pass
    def End(self) -> None:
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()