import os
import random
from Sensation6.config import config
from torch.nn import (
    Conv1d,BatchNorm1d,Module,ConstantPad1d,ModuleList,AvgPool1d,MaxPool1d,
    Linear,Dropout,LayerNorm,
    TransformerDecoderLayer,TransformerDecoder,
)
import torch
from typing import Union,Tuple
from Sensation6.sensation_models import KikitoriEncoder,KikitoriDecoder,Decoder
import math

class CausalConv1d(Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.pad = ConstantPad1d(((kernel_size-1)*dilation,0),0.0)
        self.conv = Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,padding_mode)

    def forward(self,x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class DilatedCausalConv1d(Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:Union[int,Tuple[int]],
        num_layers:int = 4,
        diration_rate:int =2,
        divs:int =4) -> None:
        super().__init__()

        dirations = [diration_rate**i for i in range(num_layers)]

        _cc = out_channels // divs
        self.init_conv_r = Conv1d(in_channels,out_channels,1)
        self.init_norm_r = BatchNorm1d(out_channels)
        self.init_conv_c = Conv1d(in_channels,_cc,1)
        self.init_norm_c = BatchNorm1d(_cc)
        self.out_conv = Conv1d(in_channels=_cc,out_channels=out_channels,kernel_size=1)
        self.out_norm = BatchNorm1d(out_channels)
        convs = []
        for i in dirations:
            convs.extend([
                CausalConv1d(_cc,_cc,kernel_size,stride=1,dilation=i),
                BatchNorm1d(_cc),
                torch.nn.ReLU(),
            ])
        self.convs = ModuleList(convs)

    def forward(self,x):
        x_o = self.init_norm_r(self.init_conv_r(x))
        x_c = self.init_norm_c(self.init_conv_c(x))
        for l in self.convs:
            x_c = l(x_c)
        x_relu = torch.relu(x_c)
        x_sigm = torch.sigmoid(x_c)
        x_c = torch.mul(x_relu,x_sigm)
        x_c = self.out_norm(self.out_conv(x_c))
        x = torch.relu(torch.add(x_o,x_c))
        return x

class Oshaberi(Module):
    def __init__(self):
        super().__init__()
        self.reset_seed()
        elem = math.prod(KikitoriEncoder.output_size)

        self.input_size = (1,1,elem*int(config.speak_second/config.recognize_second))
        self.output_size = (1,1,config.speak_length)

        # Model layers
        self.pool = AvgPool1d(2,2)
        self.decoder = KikitoriDecoder()
        self._inp = self.decoder.insize
        self.Conv1 = DilatedCausalConv1d(1,16,7)
        self.Conv2 = DilatedCausalConv1d(16,32,7)
        self.Conv3 = DilatedCausalConv1d(32,64,3)
        self.Conv4 = DilatedCausalConv1d(64,128,3)

        self.conv = Conv1d(128,1,1)

    def forward(self,x):
        x_len = x.size(0)
        x = x.view(self._inp)
        x = self.decoder(x).view(x_len,1,-1)
        x = self.pool(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = torch.tanh(self.conv(x))
        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


# Text generator -----------------------------------
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

class TextGenerator(Module):
    input_current = (1,config.generate_max_words,config.word_dim)
    input_memory = (1,config.use_mem_len,Decoder.input_size[-1])
    output_size = (1,config.vocab_size)

    incurrent = (-1,*input_current[1:])
    inmemory = (-1,input_memory[1:])
    
    nlayers = 8
    nhid = 512
    nhead = 4

    def __init__(self):
        super().__init__()
        self.reset_seed()

        # Model layers
        self.textdecoder = Decoder()
        
        self.pos_encoder = PositionalEncoding(config.word_dim,max_len=config.generate_max_words)
        decoder_layer = TransformerDecoderLayer(config.word_dim,self.nhead,dim_feedforward=self.nhid)
        self.transformerdecoder = TransformerDecoder(decoder_layer,self.nlayers)
        self.dense = Linear(config.generate_max_words*config.word_dim,8*config.generate_max_words)
        self.norm = LayerNorm(8*config.generate_max_words)
        self.fc = Linear(8*config.generate_max_words,config.vocab_size)

    def forward(self,current,memory):
        memory = self.textdecoder(memory).reshape(-1,config.text_seq_len*config.use_mem_len,config.word_dim).permute(1,0,2)
        
        pad = self.get_pad_mask(current).to(current.device)
        current = current.permute(1,0,2)
        current = self.pos_encoder(current)
        current = self.transformerdecoder(current,memory,tgt_key_padding_mask=pad).permute(1,0,2).reshape(-1,config.generate_max_words*config.word_dim)
        current = torch.relu(self.norm(self.dense(current)))
        current = self.fc(current)

        return current
        

    def get_pad_mask(self,x,pad_num:float=0.0):
        """
        x       : (N,L,E)
        output  : (N,L)
        """
        return torch.sum(x!=pad_num,axis=2) == 0

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    from torchsummaryX import summary
    model = Oshaberi()
    dummy = torch.randn(model.input_size)
    print(summary(model,dummy))