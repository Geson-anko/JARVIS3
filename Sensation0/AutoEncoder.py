from .config import config
import torch
from torch.nn import (
    Conv2d,BatchNorm2d,MaxPool2d,#AvgPool2d,
    ConvTranspose2d,Upsample,
    Module
)
import os
import random
from typing import Tuple,Union

class ResBlock2d(torch.nn.Module):
    def __init__(
        self,in_channels:int,out_channels:int,
        kernel_size_1st:Union[int,Tuple[int,int]],kernel_size_2nd:Union[int,Tuple[int,int]]) -> None:
        super().__init__()

        f_ch = int(out_channels//2)
        self.convf = Conv2d(in_channels,f_ch,kernel_size=kernel_size_1st)
        self.normf = BatchNorm2d(f_ch)
        self.pool = MaxPool2d(2)
        
        # fork1
        self.conv_ch = Conv2d(f_ch,out_channels,1)
        self.norm_ch = BatchNorm2d(out_channels)

        # fork2
        pad = torch.floor(torch.tensor(kernel_size_2nd)/2).type(torch.long)
        if pad.dim() == 0:
            pad = int(pad)
        else:
            pad = tuple(pad)
        s_ch = int(out_channels//4)
        self.conv_rc1 = Conv2d(f_ch,s_ch,1)
        self.norm_rc1 = BatchNorm2d(s_ch)
        self.conv_r = Conv2d(s_ch,s_ch,kernel_size_2nd,padding=pad)
        self.norm_r = BatchNorm2d(s_ch)
        self.conv_rc2 = Conv2d(s_ch,out_channels,1)
        self.norm_rc2 = BatchNorm2d(out_channels)

        self.norm = BatchNorm2d(out_channels)

    def forward(self,x):
        x = self.normf(self.convf(x))
        x = torch.relu(x)
        x = self.pool(x)

        # fork1
        x1 = self.norm_ch(self.conv_ch(x))

        # fork2
        x2 = self.norm_rc1(self.conv_rc1(x))
        x2 = torch.relu(x2)
        x2 = self.norm_r(self.conv_r(x2)) 
        x2 = torch.relu(x2)
        x2 = self.norm_rc2(self.conv_rc2(x2))
        
        # merge
        x = self.norm(torch.add(x1,x2))
        x = torch.relu(x)

        return x

class InverseBlock2d(Module):
    def __init__(
        self,in_channels:int,out_channels:int,
        kernel_size_1st:Union[int,Tuple[int,int]],kernel_size_2nd:Union[int,Tuple[int,int]]) -> None:
        super().__init__()
        
        self.upsampler = Upsample(scale_factor=2)
        self.Dcon1 = ConvTranspose2d(in_channels,out_channels,kernel_size_1st)
        self.norm1 = BatchNorm2d(out_channels)
        self.Dcon2 = ConvTranspose2d(out_channels,out_channels,kernel_size_2nd)
        self.norm2 = BatchNorm2d(out_channels)
    
    def forward(self,x):
        x = torch.relu(self.norm1(self.Dcon1(x)))
        x = self.upsampler(x)
        x = self.norm2(self.Dcon2(x))
        return x 

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_seed()
        self.input_size = (1,config.channels,config.width,config.height)
        self.output_size = (1,256,7,3)

        ## Model layers
        #self.pool = AvgPool2d(2)
        self.Conv1 = ResBlock2d(config.channels,8,9,5)
        self.Conv2 = ResBlock2d(8,16,7,3)
        self.Conv3 = ResBlock2d(16,32,3,3)
        self.Conv4 = ResBlock2d(32,64,3,3)
        self.Conv5 = ResBlock2d(64,128,3,3)
        self.Conv6 = ResBlock2d(128,256,3,3)
        self.outconv = Conv2d(256,256,1,1)

    def forward(self,x):
        x = x.view(-1,config.channels,config.width,config.height)
        #x = self.pool(x)
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.Conv3(x)
        x = self.Conv4(x)
        x = self.Conv5(x)
        x = self.Conv6(x)
        x = torch.tanh(self.outconv(x))

        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

class Decoder(Module):
    def __init__(self):
        super().__init__()
        SE = Encoder()
        SE.reset_seed()
        self.input_size = SE.output_size
        self.output_size = SE.input_size
        self._insize = (-1,) + SE.output_size[1:]
        # Model layers

        self.Dcon1 = InverseBlock2d(256,128,(7,3),5)
        self.Dcon2 = InverseBlock2d(128,64,(6,5),3)
        self.Dcon3 = InverseBlock2d(64,32,3,3)
        self.Dcon4 = InverseBlock2d(32,16,3,(7,5))
        self.Dcon5 = InverseBlock2d(16,3,7,(9,5))

    def forward(self,x):
        x = x.view(self._insize)
        x = self.Dcon1(x)
        x = torch.relu(x)
        x = self.Dcon2(x)
        x = torch.relu(x)
        x = self.Dcon3(x)
        x = torch.relu(x)
        x = self.Dcon4(x)
        x = torch.relu(x)
        x = self.Dcon5(x)
        x = torch.sigmoid(x)
        return x

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
        self.input_size = self.encoder.input_size

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    from torchsummaryX import summary
    from torch.utils.tensorboard import SummaryWriter
    model = AutoEncoder()
    dummy = torch.randn(model.input_size)
    print(summary(model,dummy))

    #writer = SummaryWriter()
    #writer.add_graph(model,dummy)
    #writer.close()
    
