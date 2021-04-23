# ------ your setttings ------
sensation = 'Sensation1' # your sensation folder name
initialize = True # Whether to initialize or not after debugging. If True, this code deletes Data_folder and Temp_folder
debug_time = 60 # debugging time [second]
device = 'cuda' # debugging device
# ------ end of settings -----


# ------ There is no need to change anything below this point. -----
from importlib import import_module
from MasterConfig import Config
from multiprocessing import Process,Value
from MemoryManager import MemoryManager
from send2trash import send2trash
import time
import os

if __name__ == '__main__':
    sens = import_module(sensation).Sensation(device,True)
    sens.ViewCurrentLength = True
    mm = MemoryManager(f'SensationDebug ({sensation})')
    if initialize:
        removedirs = [
            sens.Temp_folder,
            sens.Data_folder,
        ]
        removefiles = [os.path.join(i,q) for i in removedirs for q in os.listdir(i)]
        removefiles.append(Config.memory_file_form.format(sens.MemoryFormat))
        for i in removefiles:
            send2trash(i)
        mm.log('initialized')

    shutdown = Value('i',False)
    sleep = Value('i',False)
    switch = Value('i',True)
    clock = Value('d',0)
    sleepiness = Value('d',0)
    isActive = Value('i',False)
    roi = mm.create_shared_memory((sens.ReadOutLength),dtype=Config.ID_dtype,initialize=Config.init_id)
    rom = mm.create_shared_memory(
        (sens.ReadOutLength,sens.MemorySize),
        dtype=sens.dtype,initialize=0.0)
    memlist = mm.create_shared_memory((sens.MemoryListLength,),dtype=Config.ID_dtype,initialize=Config.init_id)
    newid = Value('Q')

    args = (shutdown,sleep,switch,clock,sleepiness,roi,rom,memlist,newid,isActive)
    p = Process(target=sens,args=args)
    p.start()
    mm.log('process submit')
    time.sleep(debug_time)
    shutdown.value = True
    p.join()
    mm.log('process result')
    print('NewestId',newid.value)
    print('Process Clock [second]',clock.value)
