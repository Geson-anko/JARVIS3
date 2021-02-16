import os
import torch
import os
import random
from torch.nn import(
    Module,Linear,LayerNorm
)
import math
from .AutoEncoder import Encoder

class DeltaT(Module):
    def __init__(self):
        super().__init__()
        self.reset_seed()
        self.elem = math.prod(Encoder().output_size)
        self.input_size = (1,self.elem)
        self.output_size = (1,1)

        ## Model layers
        self.dense1 = Linear(self.elem,512)
        self.norm1= LayerNorm(512)
        self.dense2 = Linear(1024,256)
        self.norm2 = LayerNorm(256)
        self.dense3 = Linear(256,1)
    
    def forward(self,x1,x2):
        x1,x2 = x1.unsqueeze(1),x2.unsqueeze(1)
        x = torch.cat([x1,x2],dim=1)
        print(x.shape)
        x = torch.relu(self.norm1(self.dense1(x)))
        x = x.view(x.size(0),-1)
        x = torch.relu(self.norm2(self.dense2(x)))
        x = torch.relu(self.dense3(x))
        return x

    def reset_seed(self,seed=0):
        os.environ['PYTHONHASHSEED'] = '0'
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    from torchsummaryX import summary
    model = DeltaT()
    dummy = torch.randn(model.input_size)
    print(summary(model,dummy,dummy))