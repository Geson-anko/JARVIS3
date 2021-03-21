import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import torch
import numpy as np

from OutputBase import OutputBase
from .config import config

from pynput import keyboard,mouse

from Sensation4 import Sensation

class Output(OutputBase):
    LogTitle:str = 'MouseOutput'
    UsingMemoryFormat:str = '4'
    MaxMemoryLength:int = 1

    SleepWaitTime:float = 1
    MaxFrameRate:int = Sensation.frame_rate

    dtype = np.float32
    torchdtype = torch.float32

    monitor_width = Sensation.monitor_width
    monitor_height = Sensation.monitor_height
    force_stop_hotkeys:str = '<ctrl>+<alt>+s'
    moving:bool = False

    def Start(self) -> None:
        self.hotkey = keyboard.HotKey(keyboard.HotKey.parse(self.force_stop_hotkeys),self.KeyActivate)
        self.lis=  keyboard.Listener(on_press=self.for_canonical(self.hotkey.press),on_release=self.for_canonical(self.hotkey.release))
        self.lis.start()
        self.log('Moving',self.moving)
        
        self.buttons = [*mouse.Button]
        self.controller = mouse.Controller()
        self.pressed_buttons = np.zeros((len(self.buttons,)),dtype=bool)

    def Update(self, MemoryData: torch.Tensor) -> None:
        print('MemoryData',MemoryData)
        if not self.moving:
            return
        data = MemoryData.view(-1).to('cpu').detach().numpy()
        x,y = data[Sensation.x_idx],data[Sensation.y_idx]
        dx,dy = data[Sensation.dx_idx],data[Sensation.dy_idx]
        scroll_dx,scroll_dy = data[Sensation.scroll_dx_idx].astype(int),data[Sensation.scroll_dy_idx].astype(int)
        buttons_info = data[Sensation.button_idx:].astype(int)
        
        x = round(x*self.monitor_width)
        y = round(y*self.monitor_height)
        dx = round((dx/self.MaxFrameRate)*self.monitor_width)
        dy = round((dy*self.MaxFrameRate)*self.monitor_height)
        
        self.controller.move(dx,dy)
        for idx,(button,press) in enumerate(zip(self.buttons,buttons_info)):
            if button is mouse.Button.unknown:
                continue
            if press and not self.pressed_buttons[idx]:
                self.controller.position=(x,y)
                self.controller.press(button)
                self.pressed_buttons[idx] = True
            elif not press and self.pressed_buttons[idx]:
                self.controller.release(button)
                self.pressed_buttons[idx] = False
        if scroll_dx != 0 and scroll_dy != 0:
            self.controller.position=(x,y)
            self.controller.scroll(scroll_dx,scroll_dy)
    
    def KeyActivate(self):
        self.moving = bool(-1 * self.moving + 1)
        self.log('Moving',self.moving)
    
    def for_canonical(self,f):
        return lambda k:f(self.lis.canonical(k))
