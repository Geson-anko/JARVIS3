import torch
from multiprocessing import Process,freeze_support

from torch._C import device

class test:
    def __init__(self) -> None:
        self.tensor = torch.randn(100,device='cuda')
    def ready(self):
        self.tensor = torch.randn(100,device='cuda')
    def run(self):
        self.ready()
        out = torch.sum(self.tensor)
        print(out.shape)

if __name__ == '__main__':
    freeze_support()
    p = Process(target=test().run)
    p.start()
    p.join()

    