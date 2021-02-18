import sys
import os
from Sensation0.sensation import Sensation
from multiprocessing import Value,Process
#sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from MasterConfig import Config as mconf
from concurrent.futures import ProcessPoolExecutor
import time
if __name__ == '__main__':
    sens = Sensation('cpu')
    
    cmd = Value('i',mconf.wake)
    switch = Value('i',True)
    clock = Value('d',0)
    sleep = Value('d',0)
    roi = sens.create_shared_memory((sens.ReadOutLength),dtype='int64',initialize=-1)
    rom = sens.create_shared_memory(
        (sens.ReadOutLength,sens.MemorySize),
        dtype='float32',initialize=0.0)
    memlist = sens.create_shared_memory((sens.MemoryListLength,),dtype='int64',initialize=-1)
    newid = Value('i')

    args = (cmd,switch,clock,sleep,roi,rom,memlist,newid)

    #executer =ProcessPoolExecutor()
    #proc = executer.submit(sens,*args)
    p = Process(target=sens,args=args)
    p.start()
    print('process submit')
    time.sleep(10)
    #cmd.value = mconf.force_sleep
    time.sleep(30)
    cmd.value = mconf.shutdown
    p.join()
    #executer.shutdown(True)
    #proc.result()
    print('process result')