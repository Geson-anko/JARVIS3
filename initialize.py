from Sensation2.sensation import Sensation
import os
from send2trash import send2trash
from debug_tools import Debug
debug= Debug('Initialization')
"""
This program initialize J.A.R.V.I.S., so please delete this before Release.
"""
from MasterConfig import Config as mconf
from MemorySearch.config import config as searchconf
from SleepManager.config import config as sleepconf
import Sensation1
import Sensation2
import Sensation3
import Sensation4


removesdirs = [
    mconf.memory_folder,
    sleepconf.temp_folder,
    Sensation1.Sensation.Temp_folder,
    Sensation1.Sensation.Data_folder,
    Sensation2.Sensation.Temp_folder,
    Sensation2.Sensation.Data_folder,
    Sensation3.Sensation.Temp_folder,
    Sensation3.Sensation.KikitoriData_folder,
    Sensation4.Sensation.Temp_folder,
]
removefiles = [
    searchconf.dict_file,
    searchconf.tempmem_file,
]


removefiles+= [os.path.join(i,q) for i in removesdirs for q in os.listdir(i)]
removefiles = [i for i in removefiles if os.path.exists(i)]
debug.log(removefiles)

while True:
    ans = input('These files will be removed. really?\nProceed ([y]/n)?').lower()
    if ans == 'y':
        for i in removefiles:
            send2trash(i)
        debug.log('removed.')
        break
    elif ans == 'n':
        debug.log('canceled.')
        break
    else:
        print("sorry, unknow answer... please enter 'y' or 'n'.")
