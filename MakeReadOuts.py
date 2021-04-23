from MemoryManager import MemoryManager
import torch
# ------ your setttings ------
sensation = 'Sensation7' # your sensation folder name
initialize = False # Whether to initialize or not before debugging. If True, this code deletes Data_folder and Temp_folder
device = 'cuda' # debugging device
mm = MemoryManager(f'MakeReadOuts ({sensation})')

# --your libs ----
import sentencepiece as spm
from Sensation7 import Sensation
# definding data loader
class DataLoader:
    def __init__(self) -> None:
        with open('Sensation7/corpus_kaburi1_mfcc2_96.txt','r',encoding='utf-8') as f:
            self.corpus = f.read().split('\n')

        self.separator = spm.SentencePieceProcessor(Sensation.Separator_params)
        self.bos = self.separator.IdToPiece(self.separator.bos_id())
        self.eos = self.separator.IdToPiece(self.separator.eos_id())
        self.modelFT = mm.load_python_obj(Sensation.FastText_params)
        self._len = len(self.corpus)
    def __len__(self):
        return self._len
    
    def __getitem__(self,idx):
        text = self.corpus[idx]
        pieces = [self.bos,*self.separator.EncodeAsPieces(text),self.eos]
        vectors = [self.modelFT.wv[i] for i in pieces if i in self.modelFT.wv]
        vectors =torch.from_numpy(np.stack(vectors).astype(Sensation.dtype))
        return vectors

# ------ end of settings -----
# --- description ---
"""
This code is to make initial ReadOutMemory and others.
"""

# ------ There is no need to change anything below this point. -----
from importlib import import_module
from MasterConfig import Config
from multiprocessing import Process,Value
from send2trash import send2trash
import time
import os
import numpy as np

release_system_cache_times = 100

if __name__ == '__main__':
    sens = import_module(sensation).Sensation(device,False)
    # initialize
    if initialize:
        removedirs = [
            sens.Temp_folder,
            sens.Data_folder,
        ]
        removefiles = [os.path.join(i,q) for i in removedirs for q in os.listdir(i)]
        removefiles.append(Config.memory_file_form.format(sens.MemoryFormat))
        for i in removefiles:
            send2trash(i)
        mm.log('initialized')

    # set class values
    sens.shutdown = Value('i',False)
    sens.sleep = Value('i',False)
    sens.switch = Value('i',True)
    sens.clock = Value('d',0)
    sens.sleepiness = Value('d',0)
    roi = np.ndarray((sens.ReadOutLength,),dtype=Config.ID_dtype)
    roi[:]=Config.init_id
    rom = np.ndarray(
        (sens.ReadOutLength,sens.MemorySize),
        dtype=sens.dtype,
    )
    rom[:] = 0
    memlist = np.ndarray((sens.MemoryListLength,),dtype=Config.ID_dtype)
    memlist[:] = 0
    newid = Value('Q')
    sens.ReadOutId = roi
    sens.ReadOutMemory = rom
    sens.MemoryList = memlist
    sens.NewestId = newid
    sens.LoadModels()
    sens.LoadPreviousValues()
    sens.DataSaving = False
    mm.log('setted class values')
    # set data loader
    data_loader = DataLoader()
    for i in range(len(data_loader)):
        indata = data_loader[i]
        print(f'\rProgress {sens.current_length/sens.KeepLength*100:3.2f} %',end='')
        if sens.current_length > sens.KeepLength:
            print('finished')
            break
        sens.MemoryProcess(indata)
        if 0 == (i%release_system_cache_times):
            mm.release_system_memory()
    sens.SaveMemories()
    sens.SavePreviousValues()


