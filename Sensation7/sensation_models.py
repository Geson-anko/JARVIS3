import torch
import torch.nn as nn
from torch.nn import (
    Module, 
    TransformerEncoder,TransformerEncoderLayer,Linear,Dropout,
    ModuleList,
)
import os
import random
import math
import copy

from .config import config
from .Conformer import Conformer
class PositionalEncoding(Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Encoder(Module):
    input_size = (1,config.text_seq_len,config.word_dim) # input size
    output_size = (1,32) # output size
    insize = (-1,*input_size[1:])

    nlayers = 6
    nhid = 512
    nhead = 4
    def __init__(self):
        super().__init__()
        self.reset_seed()

        # Model layers
        self.pos_encoder = PositionalEncoding(config.word_dim,max_len=config.text_seq_len)
        elayer = TransformerEncoderLayer(config.word_dim,self.nhead,dim_feedforward=self.nhid)
        self.transformerencoder = TransformerEncoder(elayer,self.nlayers)
        self.dense = Linear(64*8,32)


    def forward(self,x):
        x = x.view(self.insize) # x: (N,L,E)
        xlen = x.size(0)
        x = x.permute(1,0,2)
        x = self.pos_encoder(x)
        x = self.transformerencoder(x).permute(1,0,2).reshape(xlen,-1)
        x = torch.tanh(self.dense(x))
        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
class Decoder(Module):
    input_size = Encoder.output_size
    output_size = Encoder.input_size
    insize = (-1,*input_size[1:])
    
    einsize = Encoder.insize
    nlayers = 6
    nhid = 512
    nhead = 4
    def __init__(self):
        super().__init__()
        self.reset_seed()

        # Model layers
        self.dense = Linear(32,64*8)
        elayer = TransformerEncoderLayer(config.word_dim,self.nhead,dim_feedforward=self.nhid)
        self.transformerencoder = TransformerEncoder(elayer,self.nlayers)

    def forward(self,x):
        x = x.view(self.insize)
        x = torch.relu(self.dense(x)).view(self.einsize).permute(1,0,2)
        x = self.transformerencoder(x).permute(1,0,2)

        return x
    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

class AutoEncoder(Module):
    def __init__(self):
        """
        This class is used training only. 
        How about using it like this?
            >>> model = AutoEncoder()
            >>> # -- Training Model Process --
            >>> torch.save(model.encoder.state_dict(),encoder_name)
            >>> torch.save(model.decoder.state_dict(),decoder_name)
        """
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class HumanCheck(Module):
    ninp = config.recognize_length
    seq_len = config.seq_len
    input_size = (1,seq_len,ninp)
    insize = (-1,*input_size[1:])
    def __init__(self,ninp=config.recognize_length,seq_len=config.seq_len,conf_layers=4,confhid=1024,conf_dropout=0.1,nhead=4,kernel_size=5):
        super().__init__()
        self.reset_seed()
        self.ninp = ninp
        self.seq_len = seq_len
        
        self.input_size = (1,seq_len,ninp)
        self.output_size = (1,1)

        # Model layers
        conformer = Conformer(
            d_model=ninp,
            n_head=nhead,
            ff1_hsize=confhid,
            ff2_hsize=confhid,
            ff1_dropout=conf_dropout,
            conv_dropout=conf_dropout,
            ff2_dropout=conf_dropout,
            mha_dropout=conf_dropout,
            kernel_size=kernel_size,
        )
        self.conformers = ModuleList([copy.deepcopy(conformer) for _ in range(conf_layers)])
        self.dense = Linear(seq_len*ninp,1)
        

    def forward(self,x):
        x = x.view(self.insize)
        for l in self.conformers:
            x = l(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
""" Documentation


"""