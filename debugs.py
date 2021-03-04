import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Value
from tkinter.constants import S
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from MasterConfig import Config as mconf
from Sensation0.sensation import Sensation
from SleepManager import SleepManager
from GUI_controller import Controller
if __name__ == '__main__':
    
    """
    shutdown = Value('i',False)
    num = 10
    switches = [Value('i',True) for _ in range(num)]
    titles = [f'switch {i}' for i in range(num)]
    switch_objs = list(zip(titles,switches))
    con = Controler(True)
    args = (shutdown,switch_objs)
    p = Process(target=con,args=args)
    p.start() 
    time.sleep(20)
    p.join()
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
    sleep = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    newids = [Value('i',i) for i in range(4)]
    memlists = [
        sp.create_shared_memory(
            (100,),dtype='int64',initialize=mconf.init_id) for _ in range(4)
        ]
    TM = sp.create_shared_memory((500,),'int64',mconf.init_id)
    args = (shutdown,sleep,clock,sleepiness,TM,memlists,newids)
    p = Process(target=sp,args=args)
    print('process start')
    p.start()
    time.sleep(10)
    shutdown.value = True
    p.join()
    print('clock',clock.value)
    print('process end')
    """
    """
    from Sensation0.train import Train  
    train = Train('cpu',True)
    shutdown = Value('i',True)
    sleep = Value('i',True)
    train(shutdown,sleep)
    """
    """
    sens = Sensation('cpu')
    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    roi = sens.create_shared_memory((sens.ReadOutLength),dtype='int64',initialize=-1)
    rom = sens.create_shared_memory(
        (sens.ReadOutLength,sens.MemorySize),
        dtype='float32',initialize=0.0)
    memlist = sens.create_shared_memory((sens.MemoryListLength,),dtype='int64',initialize=-1)
    newid = Value('i')

    args = (shutdown,sleep,switch,clock,sleepiness,roi,rom,memlist,newid)

    #executer =ProcessPoolExecutor()
    #proc = executer.submit(sens,*args)
    p = Process(target=sens,args=args)
    p.start()
    print('process submit')
    time.sleep(15)
    #cmd.value = mconf.force_sleep
    shutdown.value = True
    p.join()
    #executer.shutdown(True)
    #proc.result()
    print('process result')
    """

    pass
