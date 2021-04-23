# ------ your settings ------
output = 'OutputOshaberi'
debug_time = 20
device='cuda'
# --- end of your settings --



# ------ There is no need to change anything below this point. -----
from importlib import import_module
from MasterConfig import Config as mconf
from multiprocessing import Process,Value
from MemoryManager import MemoryManager
import numpy as np
import time
import os

if __name__ == '__main__':
    out_proc = import_module(output).Output(device,True)
    mm = MemoryManager(f'OutputDebug ({output})')
    usingsensation = f'Sensation{out_proc.UsingMemoryFormat}'
    sense = import_module(usingsensation).Sensation
    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    rolength = out_proc.UseMemoryLowerLimit
    memsize = sense.MemorySize
    rois = [mm.create_shared_memory((rolength,),dtype=mconf.ID_dtype,initialize=mconf.init_id)]
    rois_view = [mm.inherit_shared_memory(i) for i in rois]
    rois_view[0][:] = np.arange(rolength)
    out_proc.UsingMemoryFormat = '0'
    roms = [mm.create_shared_memory((rolength,memsize),dtype=out_proc.dtype,initialize=0)]
    TempMem = mm.create_shared_memory((rolength,),dtype=mconf.ID_dtype)
    TempMem_view = mm.inherit_shared_memory(TempMem)
    TempMem_view[:] = np.arange(rolength)
    isActives = [Value('i',True) for i in range(1)]
    args = (shutdown,sleep,switch,clock,sleepiness,TempMem,rois,roms,isActives)
    p = Process(target=out_proc,args=args)
    p.start()
    mm.log('process submit')
    time.sleep(debug_time)
    shutdown.value = True
    p.join()
    mm.log('process end')
    print('Process Clock [second]',clock.value)