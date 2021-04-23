# ------ your setttings ------
TrainModule = 'OutputOshaberi' # your sensation folder name
device = 'cuda' # debugging device
# ------ end of settings -----

if __name__ == '__main__':
    from importlib import import_module
    from multiprocessing import Value
    module = import_module(TrainModule)
    func = module.Train(device,True)
    shutdown = Value('i',False)
    sleep = Value('i',True)
    func(shutdown,sleep)