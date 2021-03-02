import os
from os.path import isfile
from os.path import isdir
import pandas as pd
import h5py

configfile = 'config.csv'
# files exist check =================================================
print('config file loading.')
config = pd.read_csv(configfile,dtype=str,encoding='utf-8')
mem_head_folder = config['0'][0]
brunch = int(config['0'][1])
depth = int(config['0'][2])
# important dir
dirs = ['dict','temp','AutoEncoders']
for i in dirs:
    if isdir(i):
        print(f'{i} : folder --> ok')
    else:
        raise Exception(f'{i} : folder not exists')

# important files
files =['Memory_manager.py',
        'Memory_search.py',
        'AutoTrain.py',
        'dict/Update_dictionary.py',
        'dict/__init__.py',
        'AutoEncoders/Fit.py',
        'AutoEncoders/__init__.py']
for i in files:
    if isfile(i):
        print(f'{i} : file --> ok')
    else:
        raise Exception(f'{i} : file not exist')

# check dirs or create
from Memory_manager import Memory_manager as MM 
mm = MM()
named = 'archives/dict/{0}/'
namet = 'archives/AutoEncoder/{0}/'
for i in mm.char:
    name = named.format(i)
    if isdir(name):
        print(name+':exists')
    else:
        os.makedirs(name)
    name = namet.format(i)
    if isdir(name):
        print(name+':exists')
    else:
        os.makedirs(name)

if isdir(mem_head_folder):
    print(mem_head_folder+':exists')
else:
    os.makedirs(mem_head_folder)

#memfile = mem_head_folder+'/{}.h5'
#memfile = mm.getabspath(memfile)
#for i in mm.char:
#    name = memfile.format(i)
#    if isfile(name):
#        print(f'{name} --> exists.')
#    else:
#        with h5py.File(name,mode='a') as f:
#            pass
#============================================================


import wx
class command_blocks:
    def __init__(self,cmds,bru,dep,pipesen):
        # set command text
        self.cmd = cmds
        self.cmdhead = '<cmd>'
        self.cmdheadlen = len(self.cmdhead)
        self.shutdown = 'shutdown'
        self.sleep = 'sleep'
        self.wake = 'wake'
        self.brunch = bru
        self.depth= dep
        self.brunchcmd = 'brunch'
        self.depthcmd  = 'depth'
        self.bclen = len(self.brunchcmd)
        self.dclen = len(self.depthcmd)
        self.pipesen = pipesen

    def receive(self,text):
        head = text[:self.cmdheadlen]
        if head == self.cmdhead:
            message = text[self.cmdheadlen:]
            if message == self.shutdown:
                self.cmd.value = 999
            elif message == self.sleep:
                self.cmd.value = 1
            elif message == self.wake:
                self.cmd.value = 0
            elif message[:self.dclen] == self.depthcmd:
                try:
                    v = int(message[self.dclen:])
                    self.depth.value = v
                except ValueError:
                    print('cannot convert to int!')
            elif message[:self.bclen] == self.brunchcmd:
                try:
                    v = int(message[self.bclen:])
                    self.brunch.value = v
                except ValueError:
                    print('cannot convert to int!')
            else:
                print('Unknow command input!')

        
        self.pipesen[0] = text

class GUIinput:
    
    def __init__(self,cmds,cmdreceive,maxlength):
        self.cmdreceive = cmdreceive
        self.cmd = cmds
        self.frame = wx.Frame(None, -1, "textbox",size=(440,500))
        self.frame.SetTitle('J.A.R.V.I.S.')
        panel_ui = wx.Panel(self.frame, -1, pos=(50, 50), size=(100, 50)) 

        # message
        self.frame.label = wx.StaticText(panel_ui, -1, 'Write some message here', pos=(10, 10))
        # text box
        
        self.frame.box = wx.TextCtrl(panel_ui, -1,size=(400,200), pos=(10, 30),style=wx.TE_MULTILINE)
        self.frame.box.Bind(wx.EVT_TEXT_MAXLEN, self.Clicked) #text enter
        self.frame.box.SetMaxLength(maxlength)

        # creare
        #crearUI = wx.Panel(self.frame,-1,pos=(50,100),size=(100,50))
        cre = wx.Button(panel_ui,-1,'GO!',pos=(10,230))
        cre.Bind(wx.EVT_BUTTON,self.Clicked)
        self.frame.Show(True)

    def Clicked(self,event):
        text = self.frame.box.GetValue()
        self.cmdreceive(text)
        if self.cmd.value == 999:
            self.frame.Close(True)
        else:
            self.frame.box.Clear()


# calling shared memory valuables ====================
import numpy as np
import math
from multiprocessing import Value,Process,shared_memory,freeze_support
cmd = Value('i',0)
brunch = Value('i',brunch)
depth = Value('i',depth)
shared_mem_lists = []
# input text 
maxlength = int(config['0'][7])
dummy = np.array(['a'*maxlength])
size = dummy.itemsize
pubmem = shared_memory.SharedMemory(create=True,size=size)
textpipe = np.ndarray(shape=(1,),dtype=dummy.dtype,buffer=pubmem.buf)
textpipe[:] = ''
textpipes = (textpipe,pubmem)
shared_mem_lists.append(pubmem)

# call memlist
names = ['mem{}'.format(i) for i in range(2)]
arrs = []

dummyID = config['0'][5]
mem_listlim = int(config['0'][6])
dummy =[dummyID]*mem_listlim
dummy = np.array(dummy)
dumsha = dummy.shape
size = math.prod(dumsha) * dummy.itemsize
for i in names:
    pubmem = shared_memory.SharedMemory(create=True,size=size)
    arr = np.ndarray(shape=dumsha,dtype=dummy.dtype,buffer=pubmem.buf)
    arr[:] = ''
    shared_mem_lists.append(pubmem)
    arrs.append((arr,pubmem))

mem_list0,mem_list1 = arrs[0],arrs[1]

# call newest mems
dummy = np.array([dummyID])
dumsha = dummy.shape
size = math.prod(dumsha) * dummy.itemsize
newests = []
for i in names:
    pubmem = shared_memory.SharedMemory(create=True,size=size)
    arr = np.ndarray(shape=dumsha,dtype=dummy.dtype,buffer=pubmem.buf)
    arr[:] = ''
    shared_mem_lists.append(pubmem)
    newests.append((arr,pubmem))

starttime0,starttime1 =[Value('d',0.0) for _ in names]
mem_list0,mem_list1 = arrs[0],arrs[1]
newID0,newID1 = newests[0],newests[1]

#===================================================
from sensation1 import sensation1
from AutoTrain import AutoTrain
import time
from JARVIS_logo import logo
from outputtext import outputtext
def main():
    application = wx.App()
    cmdblock = command_blocks(cmd,brunch,depth,textpipe)
    processes =[sensation1,
                outputtext,
                AutoTrain]
    inputargs =[(cmd,brunch,depth,starttime1,mem_list1,newID1,textpipes),
                (cmd,mem_list1),
                (cmd,)]
    if len(inputargs) != len(processes):
        raise Exception('Not match processes and inputargs')
    pros = [] #

    while True:
        if pros == []:#
            for pro,inp in zip(processes,inputargs):
                p = Process(target=pro,args=inp)
                p.start()
                pros.append(p)
            time.sleep(30)#
            logo()

        if cmd.value == 999:
            for i0 in pros:
                i0.join()
            break
        else:
            frame = GUIinput(cmd,cmdblock.receive,maxlength)
            application.SetTopWindow(frame.frame)
            application.MainLoop()

    for i1 in shared_mem_lists:
        i1.close()
        i1.unlink()
    
    config = pd.read_csv(configfile,encoding='utf-8')
    config['0'][1] = brunch.value
    config['0'][2] = depth.value
    config.to_csv(configfile,encoding='utf-8',index=False)
    
    print('shutdowned')


if __name__ == '__main__':
    freeze_support()
    main()
    