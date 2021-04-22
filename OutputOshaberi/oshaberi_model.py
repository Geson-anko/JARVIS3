import os
import random
from Sensation7.config import config
import torch.nn as nn
import torch
from typing import Union,Tuple
import math

class CausalConv1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.pad = nn.ConstantPad1d(((kernel_size-1)*dilation,0),0.0)
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,padding_mode)

    def forward(self,x):
        x = self.pad(x)
        x = self.conv(x)
        return x

class DilatedCausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:Union[int,Tuple[int]],
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int =4,
        dropout:float=0.1) -> None:
        super().__init__()

        dilations = [dilation_rate**i for i in range(num_layers)]

        _cc = out_channels // divs
        self.init_conv_r = nn.Conv1d(in_channels,out_channels,1)
        self.init_norm_r = nn.BatchNorm1d(out_channels)
        self.init_conv_c = nn.Conv1d(in_channels,_cc,1)
        self.init_norm_c = nn.BatchNorm1d(_cc)
        self.out_conv = nn.Conv1d(in_channels=_cc,out_channels=out_channels,kernel_size=1)
        self.out_norm = nn.BatchNorm1d(out_channels)
        convs = []
        for i in dilations:
            convs.extend([
                CausalConv1d(_cc,_cc,kernel_size,stride=1,dilation=i),
                nn.BatchNorm1d(_cc),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.convs = nn.ModuleList(convs)

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

class ConvNorm1d(nn.Module):
    def __init__(self,in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]],
                stride: Union[int, Tuple[int]] = 1, padding: Union[int, Tuple[int]] = 0, 
                dilation: Union[int, Tuple[int]]= 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias,padding_mode)
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self,x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class DilatedDepthUnit(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int =2,
        end_activation=torch.relu) -> None:
        super().__init__()
        assert (kernel_size%2==1)
        self.ef = end_activation
        
        dilations = [dilation_rate**i for i in range(num_layers)]
        pads = [int((kernel_size*i-i)/2) for i in dilations]
        _c0 = in_channels//divs
        
        
        self.fconv = ConvNorm1d(in_channels,_c0,1)
        self.cconv = ConvNorm1d(in_channels,out_channels,1)

        self.convs = nn.ModuleList([ConvNorm1d(_c0,_c0,kernel_size,dilation=dilations[i],padding=pads[i]) for i in range(num_layers)])

        self.oconv = ConvNorm1d(_c0,out_channels,1)

    def forward(self,x):
        x_fork = self.cconv(x)
        
        x_o = self.fconv(x)
        x = torch.relu(x_o)
        for l in self.convs:
            _x = l(x)
            x = torch.relu(torch.add(_x,x_o))
            x_o = _x.clone()
        x = self.oconv(x)
        x = self.ef(torch.add(x,x_fork))
        return x

class DilatedWideUnit(nn.Module):
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        kernel_size:int,
        num_layers:int = 4,
        dilation_rate:int =2,
        divs:int=2,
        end_activation=torch.relu) -> None:
        super().__init__()
        assert (kernel_size%2==1)
        assert (in_channels>1)
        self.ef = end_activation
        self.num_layers = num_layers
        
        _c0 = in_channels//divs
        dilations = [dilation_rate**i for i in range(num_layers)]
        pads = [int((kernel_size*i-i)/2) for i in dilations]
        self.fconv = ConvNorm1d(in_channels,_c0,1)
        self.cconv = ConvNorm1d(in_channels,out_channels,1)
        self.convs = nn.ModuleList([ConvNorm1d(_c0,_c0,kernel_size,dilation=dilations[i],padding=pads[i]) for i in range(num_layers)])
        self.oconv = ConvNorm1d(_c0,out_channels,1)

    def forward(self,x):
        x_fork = self.cconv(x)
        x_init = torch.relu(self.fconv(x))
        x = x_init/self.num_layers
        for l in self.convs:
            x = torch.add(l(x_init)/self.num_layers,x)
        x = torch.relu(x)
        x = torch.add(x_fork,self.oconv(x))
        x = self.ef(x)
        return x


class TacotronStyle(nn.Module):
    def __init__(self,inchannel:int,outchannels:int,scale_factor:int,kernel_size:int=5,lstm_layers:int=2,conv_layers:int=3):
        super().__init__()
        
        pad = math.floor(kernel_size/2)
        _c1 = inchannel*4
        _c2 = inchannel*8
        # layers
        self.upper = nn.Upsample(scale_factor=scale_factor)
        self.iconv = nn.Conv1d(inchannel,_c1,kernel_size,padding=pad)
        self.inorm  = nn.BatchNorm1d(_c1)
        self.lstm = nn.LSTM(_c1,_c1,lstm_layers,batch_first=True,bidirectional=True)
        layers = []
        for _ in range(conv_layers):
            layers.extend(
                [nn.Conv1d(_c2,_c2,kernel_size,padding=pad),nn.BatchNorm1d(_c2),nn.ReLU()]
            )
        self.resconv = ConvNorm1d(_c2,_c2,kernel_size,padding=pad)
        self.convs = nn.Sequential(*layers)
        self.oconv = nn.Conv1d(_c2,outchannels,kernel_size,padding=pad)
        self.onorm = nn.BatchNorm1d(outchannels)
    
    def forward(self,x):
        x = x.permute(0,2,1)
        x = torch.relu(self.inorm(self.iconv(x))).permute(0,2,1)
        x,_ = self.lstm(x)
        x = x.permute(0,2,1)
        x0 = self.upper(x)
        x = self.convs(x0)
        x = self.resconv(x)
        x = torch.relu(torch.add(x,x0))
        x = torch.relu(self.onorm(self.oconv(x)))
        return x

class WaveGlowStyle(nn.Module):
    def __init__(self,inchannels:int,conv_layers=3):
        super().__init__()
        _c1 = inchannels*2
        _c2 = _c1*4

        # Layers
        fConv = DilatedCausalConv1d(inchannels,_c1,5,8)
        upper_x2 = nn.Upsample(scale_factor=2)
        upper_x4 = nn.Upsample(scale_factor=4)
        sConv = DilatedCausalConv1d(_c1,_c2,3,4)
        Convs = [DilatedCausalConv1d(_c2,_c2,3,4) for _ in range(conv_layers)]
        uConv1 = DilatedCausalConv1d(_c2,_c1,3,4)
        uConv2 = DilatedCausalConv1d(_c1,inchannels,7,2)
        fconv = CausalConv1d(inchannels,1,3)

        self.layers = nn.Sequential(
            fConv,
            sConv,
            *Convs,
            upper_x4,
            uConv1,
            upper_x2,
            uConv2,
            fconv,
            nn.Tanh(),
        )

    def forward(self,x):
        x = self.layers(x)
        return x


class Oshaberi(nn.Module):
    input_size:tuple = (1,config.speak_seq_len,config.MFCC_channels)
    output_size:tuple = (1,1,config.speak_length)
    insize:tuple = (-1,*input_size[1:])
    def __init__(self):
        super().__init__()
        self.reset_seed()
        self.tacotron = TacotronStyle(self.input_size[-1],64,4,5,2,3)
        self.upper = nn.Upsample(scale_factor=20)
        self.waveglow = WaveGlowStyle(64,3)
    
    def forward(self,x):
        x = self.tacotron(x)
        x = self.upper(x)
        x = self.waveglow(x)
        return x
        
    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

class OshaberiMel(nn.Module):
    input_size:tuple = (1,config.speak_seq_len,config.MFCC_channels)
    output_size:tuple = (1,config.speak_mel_channels,config.speak_stft_length)
    insize:tuple = (-1,*input_size[1:])

    def __init__(self):
        super().__init__()
        self.reset_seed()
    
        self.dconv = nn.ConvTranspose1d(config.MFCC_channels,64,config.speak_stft_length-config.speak_seq_len+1)    
        self.norm = nn.BatchNorm1d(64)
        self.tacotron = TacotronStyle(64,128,1,conv_layers=2)
    
    def forward(self,x):
        x = x.view(self.insize)
        x = x.permute(0,2,1)
        x = torch.relu(self.norm(self.dconv(x))).permute(0,2,1)
        x = self.tacotron(x)
        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

import pytorch_lightning as pl
from torch.nn import functional as F
class OshaberiWave(pl.LightningModule):
    input_size:tuple = OshaberiMel.output_size
    output_size:tuple = (1,1,config.speak_length)
    insize:tuple = (-1,*input_size[1:])

    lr:float = 0.001
    def __init__(self):
        super().__init__()
        self.reset_seed()
        # layers
        conv = ConvNorm1d(self.input_size[1],128,kernel_size=2)
        upper_x20 = nn.Upsample(scale_factor=20)
        upper_x4 = nn.Upsample(scale_factor=4)
        conv0 = DilatedDepthUnit(128,256,7,6,divs=2)
        conv1 = DilatedDepthUnit(256,512,3,5,divs=2)
        conv2 = DilatedDepthUnit(512,128,7,5,divs=4)
        conv3 = DilatedDepthUnit(128,64,5,3,divs=2)
        out_conv = nn.Conv1d(64,1,3,padding=1)

        self.layers = nn.Sequential(
            conv,nn.ReLU(),
            upper_x4,
            conv0,conv1,
            upper_x4,
            conv2,
            upper_x20,
            conv3,
            out_conv,nn.Tanh()
        )
    
    def forward(self,x):
        x = x.view(self.insize)
        x = self.layers(x)
        return x
        
    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        return optimizer

    def training_step(self,train_batch,batch_idx):
        data,ans = train_batch
        out = self(data)
        loss = F.mse_loss(out,ans)
        self.log('train_loss',loss)
        return loss


from Sensation7.sensation_models import Decoder,PositionalEncoding
from torchmetrics import Accuracy
class OshaberiText(pl.LightningModule):
    input_current = (1,config.generate_max_words,config.word_dim)
    input_memory = (1,config.use_mem_len,Decoder.input_size[-1])
    output_size = (1,config.vocab_size)
    incurrent = (-1,*input_current[1:])
    inmemory = (-1,input_memory[1:])
    
    nlayers = 16
    nhid = 1024
    nhead = 8

    lr:float = 0.001
    lr_scheduler_threshold_epochs:int=16
    lr_scheduler_scale_factor:float = 1.1
    def lr_curve(self,epochs):
        if epochs < self.lr_scheduler_threshold_epochs:
            return self.lr_scheduler_scale_factor**epochs
        else:
            return self.lr

    def __init__(self):
        super().__init__()
        self.reset_seed()
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy()

        # Model layers
        self.textdecoder = Decoder()
        self.pos_encoder = PositionalEncoding(config.word_dim,max_len=config.generate_max_words)
        decoder_layer = nn.TransformerDecoderLayer(config.word_dim,self.nhead,dim_feedforward=self.nhid)
        self.transformerdecoder = nn.TransformerDecoder(decoder_layer,self.nlayers)
        self.dense = nn.Linear(config.generate_max_words*config.word_dim,8*config.generate_max_words)
        self.fc = nn.Linear(8*config.generate_max_words,config.vocab_size)

    def forward(self,current:torch.Tensor,memory:torch.Tensor)->torch.Tensor:
        memory = self.textdecoder(memory).reshape(-1,config.text_seq_len*config.use_mem_len,config.word_dim).permute(1,0,2)
        
        pad = self.get_pad_mask(current).to(current.device)
        current = current.permute(1,0,2)
        current = self.pos_encoder(current)
        current = self.transformerdecoder(current,memory,tgt_key_padding_mask=pad).permute(1,0,2).reshape(-1,config.generate_max_words*config.word_dim)
        current = torch.relu(self.dense(current))
        current = self.fc(current)
        return current

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,self.lr_curve)
        return [optimizer],[scheduler]

    def training_step(self,train_batch,batch_idx):
        cur,mem,ans = train_batch
        out = self(cur,mem)
        loss = self.criterion(out,ans)
        acc = self.accuracy(torch.argmax(out,dim=-1),ans)
        self.log('train_loss',loss)
        self.log('train_accuracy',acc)
        return loss


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
    model = OshaberiText()
    cur,mem = torch.randn(model.input_current),torch.randn(model.input_memory)
    summary(model,cur,mem)
    #raise 0
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    writer.add_graph(model,[cur,mem])
    writer.close()