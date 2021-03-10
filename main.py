from MasterConfig import Config
from MemoryManager import MemoryManager
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Value,Process
import time


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
    process_funcs,process_args = [],[]
    thread_funcs,thread_args = [],[]
    NewestIds = []
    MemoryLists = []
    TempMemoryLength = 0
    Switches = []
    Clocks = []
    mm = MemoryManager(log_title,debug_mode)
    
    ## Process Import Zone
    # sensations -------------------------
    from Sensation0 import Sensation as sense0

    # outputs ----------------------------


    # Others
    from MemorySearch import MemorySearch
    from Trainer import Train
    from GUI_controller import Controller
    from SleepManager import SleepManager

    ## Process Calling 
    # Sensations
    # sensation0 ------------------------
    device0 = 'cuda'
    debug_mode0 = False
    func = sense0(device0,debug_mode0)
    process_funcs.append(func)
    switch0 = Value('i',True)
    Switches.append((func.log_title,switch0))
    clock0 = Value('d',0.0)
    Clocks.append((func.log_title,clock0))
    
    ReadOutId0 = mm.create_shared_memory((func.ReadOutLength,),dtype=Config.ID_dtype,initialize=Config.init_id)
    ReadOutMemory0 = mm.create_shared_memory(
        (func.ReadOutLength,func.MemorySize),dtype='float16',initialize=0.0
    )
    MemoryList0 = mm.create_shared_memory((func.MemoryListLength,),dtype=Config.ID_dtype,initialize=Config.init_id)
    MemoryLists.append(MemoryList0)
    TempMemoryLength  += func.MemoryListLength
    NewestId0 = Value('i',Config.init_id)
    NewestIds.append(NewestId0)

    args = (shutdown,sleep,switch0,clock0,sleepiness,ReadOutId0,ReadOutMemory0,MemoryList0,NewestId0)
    process_args.append(args)
    mm.log('Sensation0 is ready.')


    # MemorySearch --------------------
    debug_mode_search = False
    func = MemorySearch(debug_mode_search)
    process_funcs.append(func)
    clock_search = Value('d',0.0)
    Clocks.append((func.log_title,clock_search))
    TempMemory = mm.create_shared_memory((TempMemoryLength,),dtype=Config.ID_dtype,initialize=Config.init_id)
    args = (shutdown,sleep,clock_search,sleepiness,TempMemory,MemoryLists,NewestIds)
    process_args.append(args)
    mm.log('MemorySearch is ready')

    # Outputs



    # Trainer -------------------------
    device_trainer = 'cuda'
    debug_mode_trainer = False
    func = Train(device_trainer,debug_mode_trainer)
    process_funcs.append(func)
    args = (shutdown,sleep)
    process_args.append(args)
    mm.log('Trainer is ready.')
    
    # SleepManager -------------------
    debug_mode_sleep = False
    func = SleepManager(debug_mode_sleep)
    thread_funcs.append(func)
    switch_sleep = Value('i',True)
    Switches.append((func.log_title,switch_sleep))
    clock_sleep = Value('d',0.0)
    Clocks.append((func.log_title,clock_sleep))
    args = (shutdown,sleep,switch_sleep,clock_sleep,sleepiness,NewestIds)
    thread_args.append(args)
    mm.log('SleepManager is ready')

    # Controller --------------------
    debug_mode_controller = False
    func = Controller(debug_mode_controller)
    thread_funcs.append(func)
    args = (shutdown,Switches)
    thread_args.append(args)
    mm.log('Controller is ready')



    ### Process start
    processes = []
    for (func,args) in zip(process_funcs,process_args):
        p = Process(target=func,args=args)
        p.start()
        processes.append(p)
    
    executer = ThreadPoolExecutor(max_workers=len(thread_funcs))
    for (func,args) in zip(thread_funcs,thread_args):
        executer.submit(func,*args)
    mm.log('J.A.R.V.I.S. started')
    
    time.sleep(10)
    with open(Config.logo_file,encoding='utf-8') as f:
        print(f.read())

    executer.shutdown(True)
    for p in processes:
        p.join()
    mm.destory_shared_memory()
    mm.log('J.A.R.V.I.S. shutdowned')

if __name__ == '__main__':
    main()