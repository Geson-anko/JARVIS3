import importlib

import torch
from MasterConfig import Config
from MemoryManager import MemoryManager
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value,Process,freeze_support
import time
from typing import Tuple,Any,Callable

def ProcessAfterThreading(funcs_and_args:Tuple[Tuple[Callable,Tuple[Any]],...]) -> None:
    executor = ThreadPoolExecutor()
    for (func, args) in funcs_and_args:
        executor.submit(func,*args)
    executor.shutdown(True)


def main():
    log_title = 'main'
    debug_mode = False
    """
    This is main function.
    When you add a sensation, please write code like this below.
    You have to set switch, clocks, and other shared memories along your function input arguments.
    """
    shutdown = Value('i',False)
    sleep = Value('i',False)
    sleepiness = Value('d',0)
    NewestIds = []
    MemoryLists = []
    ReadOutMemories = []
    ReadOutIds = []
    TempMemoryLength = 0
    Switches = []
    Clocks = []
    logtitles = []
    IsActives = []
    mm = MemoryManager(log_title,debug_mode)
    processes = [[] for _ in range(Config.max_processes)]
    
    ## Process Import Zone
    from MemorySearch import MemorySearch
    from Trainer import Train
    from GUI_controller import Controller
    from SleepManager import SleepManager

    for i in Config.sensation_modules:
        func = importlib.import_module(i[0])
        device = torch.device(i[1])
        group_number = i[2]
        func = func.Sensation(device,debug_mode)
        logtitles.append(func.LogTitle)
        switch = Value('i',True)
        Switches.append((func.LogTitle,switch))
        clock = Value('d',0.0)
        Clocks.append((func.LogTitle,clock))
        ReadOutId = mm.create_shared_memory((func.ReadOutLength,),dtype=Config.ID_dtype,initialize=Config.init_id)
        ReadOutIds.append(ReadOutId)
        ReadOutMemory = mm.create_shared_memory(
            (func.ReadOutLength,func.MemorySize),dtype=func.dtype,initialize=0.0
        )
        ReadOutMemories.append(ReadOutMemory)
        MemoryList = mm.create_shared_memory((func.MemoryListLength,),dtype=Config.ID_dtype,initialize=Config.init_id)
        MemoryLists.append(MemoryList)
        TempMemoryLength += func.MemoryListLength
        NewestId = Value('Q',Config.init_id)
        NewestIds.append(NewestId)
        
        isActive = Value('i',False)
        IsActives.append(isActive)
        args = (shutdown,sleep,switch,clock,sleepiness,ReadOutId,ReadOutMemory,MemoryList,NewestId,isActive)
        processes[group_number].append((func,args))
        mm.log(func.LogTitle,'is ready.')


    # MemorySearch --------------------
    debug_mode_search = False
    func = MemorySearch(debug_mode_search)
    switch_search = Value('i',True)
    Switches.append((func.LogTitle,switch_search))
    clock_search = Value('d',0.0)
    Clocks.append((func.LogTitle,clock_search))
    TempMemoryLength = int(Config.tempmem_scale_factor*TempMemoryLength)
    TempMemory = mm.create_shared_memory((TempMemoryLength,),dtype=Config.ID_dtype,initialize=Config.init_id)
    args = (shutdown,sleep,switch_search,clock_search,sleepiness,TempMemory,MemoryLists,NewestIds)
    logtitles.append(func.LogTitle)
    processes[Config.main_proceess_group_number].append((func,args))
    mm.log(func.LogTitle,'is ready.')

    # Outputs ------------------------
    for i in Config.output_modules:
        func = importlib.import_module(i[0])
        device = torch.device(i[1])
        group_number= i[2]
        func = func.Output(device,debug_mode)
        logtitles.append(func.LogTitle)
        switch = Value('i',True)
        Switches.append((func.LogTitle,switch))
        clock = Value('d',0.0)
        Clocks.append((func.LogTitle,clock))
        args = (shutdown,sleep,switch,clock,sleepiness,TempMemory,ReadOutIds,ReadOutMemories,IsActives)
        processes[group_number].append((func,args))
        mm.log(func.LogTitle,'is ready.')

    # Trainer -------------------------
    debug_mode_trainer = False
    func = Train(Config.device_trainer,debug_mode_trainer)
    args = (shutdown,sleep)
    processes[Config.main_proceess_group_number].append((func,args))
    logtitles.append(func.LogTitle)
    mm.log(func.LogTitle,'is ready.')

    # SleepManager -------------------
    debug_mode_sleep = False
    func = SleepManager(debug_mode_sleep)
    switch_sleep = Value('i',True)
    Switches.append((func.LogTitle,switch_sleep))
    clock_sleep = Value('d',0.0)
    Clocks.append((func.LogTitle,clock_sleep))
    args = (shutdown,sleep,switch_sleep,clock_sleep,sleepiness,NewestIds)
    processes[Config.main_proceess_group_number].append((func,args))

    logtitles.append(func.LogTitle)
    mm.log(func.LogTitle,'is ready.')

    # Controller --------------------
    debug_mode_controller = False
    func = Controller(debug_mode_controller)
    args = (shutdown,Switches)
    processes[Config.main_proceess_group_number].append((func,args))
    logtitles.append(func.LogTitle)
    mm.log(func.LogTitle,'is ready.')

    # checking LogTitle duplication
    logts = set()
    for i in logtitles:
        if i in logts:
            mm.exception(f'duplicated {i}')
        else:
            logts.add(i)

    ### Process start
    working_processes = []
    for pros in processes:
        p = Process(target=ProcessAfterThreading,args=[pros])
        p.start()
        working_processes.append(p)
    mm.log('J.A.R.V.I.S. started')

    time.sleep(10)
    with open(Config.logo_file,encoding='utf-8') as f:
        print(f.read())
    while not shutdown.value:
        text = 'Clocks '
        text  += '|'.join([' {0} : {1:3.3f}'.format(p,q.value) for p,q in Clocks])
        text += '| Sleepiness :{0:3.3f}'.format(sleepiness.value)
        text += f'| Sleep : {bool(sleep.value)}'
        mm.log(text)
        mm.release_system_memory()
        time.sleep(30)
    for p in working_processes:
        p.join()
    mm.destory_shared_memory()
    mm.log('J.A.R.V.I.S. shutdowned')



if __name__ == '__main__':
    freeze_support()
    main()