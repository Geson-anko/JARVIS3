import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from multiprocessing import Process, Value
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from MasterConfig import Config as mconf
from Sensation1.sensation import Sensation
from SleepManager import SleepManager
from GUI_controller import Controller
from Trainer import Train
import OutputVision
import numpy as np
if __name__ == '__main__':
    """
    from OutputOshaberi import Train
    train = Train('cuda',True)
    shutdown = Value('i',False)
    sleep = Value('i',True)
    train(shutdown,sleep)
    """
    """
    import OutputOshaberi
    from Sensation6 import Sensation
    proc = OutputOshaberi.Output('cuda',True)
    proc.UsingMemoryFormat = '0'
    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    rolength = 10
    rois = [proc.create_shared_memory((rolength,),dtype=mconf.ID_dtype,initialize=mconf.init_id)]
    rois[0][0][:] = np.arange(rolength)
    roms = [proc.create_shared_memory((rolength,Sensation.MemorySize),dtype='float16',initialize=0)]
    TempMem = proc.create_shared_memory((rolength,),dtype=mconf.ID_dtype)
    TempMem[0][:] = np.arange(rolength)
    isActives = [Value('i',True) for i in range(1)]
    args = (shutdown,sleep,switch,clock,sleepiness,TempMem,rois,roms,isActives)
    p = Process(target=proc,args=args)
    print('process submit')
    p.start()
    time.sleep(10)
    shutdown.value = True
    p.join()
    print('process ended')
    """
    """
    proc = OutputVision.Output('cuda',True)
    proc.UsingMemoryFormat = '0'
    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    rolength = 10
    rois = [proc.create_shared_memory((rolength,),dtype=mconf.ID_dtype,initialize=mconf.init_id)]
    rois[0][0][:] = np.arange(rolength)
    roms = [proc.create_shared_memory((rolength,OutputVision.output.vsense.MemorySize),dtype='float16',initialize=0)]
    TempMem = proc.create_shared_memory((rolength,),dtype=mconf.ID_dtype)
    TempMem[0][:] = np.arange(rolength)
    isActives = [Value('i',True) for i in range(1)]
    args = (shutdown,sleep,switch,clock,sleepiness,TempMem,rois,roms,isActives)
    p = Process(target=proc,args=args)
    print('process submit')
    p.start()
    time.sleep(10)
    shutdown.value = True
    p.join()
    print('process ended')
    """
    """
    #process.LoadModels()
    isActives = [Value('i',True) for i in range(10)]
    shutdown = Value('i',False)
    process.IsActives = isActives
    process.shutdown = shutdown
    #process.WaitAllTrue()
    ReadOutIds = [np.arange(10)]
    ReadOutMemories = [np.random.randn(10,3)]
    process.ReadOutId = ReadOutIds
    process.ReadOutMemory = ReadOutMemories
    process.UsingMemoryFormat = '0'
    process.MaxMemoryLength = 10
    process.SetUsingReadOuts()
    print(process.ReadOutId)
    TempMem = np.arange(5).astype('int64')
    process.TemporaryMemory = TempMem
    print(process.ReadOutMemory)
    print(process.GetMemory())
    """
    """
    from Sensation3 import Train
    train = Train('cuda',True)
    shutdown = Value('i',False)
    sleep = Value('i',True)
    train(shutdown,sleep)
    """
    """
    from Sensation1.train import Train  
    train = Train('cuda',True)
    shutdown = Value('i',False)
    sleep = Value('i',True)
    train(shutdown,sleep)
    """
    """
    sens = Sensation('cuda',True)
    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    isActive = Value('i',False)
    roi = sens.create_shared_memory((sens.ReadOutLength),dtype='int64',initialize=-1)
    rom = sens.create_shared_memory(
        (sens.ReadOutLength,sens.MemorySize),
        dtype='float16',initialize=0.0)
    memlist = sens.create_shared_memory((sens.MemoryListLength,),dtype='int64',initialize=-1)
    newid = Value('Q')

    args = (shutdown,sleep,switch,clock,sleepiness,roi,rom,memlist,newid,isActive)

    #executer =ProcessPoolExecutor()
    #proc = executer.submit(sens,*args)
    p = Process(target=sens,args=args)
    p.start()
    print('process submit')
    _roi =sens.inherit_shared_memory(roi)
    _rom = sens.inherit_shared_memory(rom)
    _mem = sens.inherit_shared_memory(memlist)
    time.sleep(30)
    #cmd.value = mconf.force_sleep
    shutdown.value = True
    p.join()
    #executer.shutdown(True)
    #proc.result()
    print('process result')
    print(_roi[:10])
    print(_rom[:10])
    print(_mem)
    print(1/clock.value)
    """
    """
    shutdown = Value('i',False)
    sleep = Value('i',True)
    train = Train('cuda',True)
    args = (shutdown,sleep)
    print('process start')
    p = Process(target=train,args=args)
    p.start()
    time.sleep(10)
    shutdown.value = True
    p.join()
    print('process end')
    """ 
    """
    shutdown = Value('i',False)
    num = 10
    switches = [Value('i',True) for _ in range(num)]
    titles = [f'switch {i}' for i in range(num)]
    switch_objs = list(zip(titles,switches))
    con = Controller(True)
    args = (shutdown,switch_objs)
    #p = Process(target=con,args=args)
    executer = ThreadPoolExecutor(1)
    executer.submit(con,*args)
    #p.start() 
    time.sleep(10)
    #p.join()
    executer.shutdown(True)
    print(shutdown.value)
    for i in switches:
        print(i.value)
    """
    """
    import matplotlib.pyplot as plt
    import numpy as np
    newids = [Value('i',i) for i in range(4)]
    SM = SleepManager(True)
    shutdown = Value('i',False)
    sleep = Value('i',False)
    clock = Value('d',0)
    switch = Value('i',True)
    sleepiness = Value('d',0)
    #x = np.linspace(0,48*60*60,1000)
    #y = [SM.sleepiness(i) for i in x]
    #plt.plot(x,y)
    #plt.show() 
    print(SM.SleepinessCurve(15*60*60+44*60+10))
    #raise Exception
    args = (shutdown,sleep,switch,clock,sleepiness,newids
    )
    p = Process(target=SM,args=args)
    p.start()
    print('pro start')
    time.sleep(4)
    print('clock',clock.value)
    print('sleep',sleepiness.value)
    shutdown.value = True
    p.join()
    print('clock',clock.value)
    print('sleep',sleepiness.value)
    print('process end')
    """
    """
    from MemorySearch import MemorySearch
    import copy
    sp = MemorySearch(True)
    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    newids = [Value('i',i) for i in range(4)]
    memlists = [
        sp.create_shared_memory(
            (5,),dtype='int64',initialize=mconf.init_id) for _ in range(4)
        ]
    _mem = [sp.inherit_shared_memory(i) for i in memlists]
    TM = sp.create_shared_memory((10,),'int64',mconf.init_id)
    args = (shutdown,sleep,switch,clock,sleepiness,TM,memlists,newids)
    p = Process(target=sp,args=args)
    print('process start')
    p.start()
    time.sleep(1)
    _mem[0][0] = 0
    _mem[0][1] = 1
    newids[0].value = 1
    time.sleep(5)
    shutdown.value = True
    p.join()
    print('clock',clock.value)
    print('process end')
    """
    """
    from Sensation1.train import Train  
    train = Train('cuda',True)
    shutdown = Value('i',False)
    sleep = Value('i',True)
    train(shutdown,sleep)
    """
    """
    sens = Sensation('cuda')
    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    roi = sens.create_shared_memory((sens.ReadOutLength),dtype='int64',initialize=-1)
    rom = sens.create_shared_memory(
        (sens.ReadOutLength,sens.MemorySize),
        dtype='float16',initialize=0.0)
    memlist = sens.create_shared_memory((sens.MemoryListLength,),dtype='int64',initialize=-1)
    newid = Value('i')

    args = (shutdown,sleep,switch,clock,sleepiness,roi,rom,memlist,newid)

    #executer =ProcessPoolExecutor()
    #proc = executer.submit(sens,*args)
    p = Process(target=sens,args=args)
    p.start()
    print('process submit')
    _roi =sens.inherit_shared_memory(roi)
    _rom = sens.inherit_shared_memory(rom)
    _mem = sens.inherit_shared_memory(memlist)
    time.sleep(15)
    #cmd.value = mconf.force_sleep
    shutdown.value = True
    p.join()
    #executer.shutdown(True)
    #proc.result()
    print('process result')
    print(_roi[:10])
    print(_rom[:10])
    print(_mem)
    """

    pass
