from multiprocessing import Process
import time
from MemoryManager import MemoryManager


mm = MemoryManager()
until = time.time() +3
def modify1(mem):
    memory = mm.inherit_shared_memory(mem)
    while time.time() < until:
        pass
    memory[:] = 1
    print('modified 1')

def modify2(mem):
    memory = mm.inherit_shared_memory(mem)
    while time.time() < until:
        pass
    memory[:] = 2
    print('modified 2')

if __name__ == '__main__':
    share = mm.create_shared_memory((1000,),dtype='int64',initialize=0)
    view = mm.inherit_shared_memory(share)
    pro = []
    for i in [modify2,modify1]:
        p = Process(target=i,args=(share,))
        p.start()
        pro.append(p)
    for i in pro:
        i.join()
    print(view)
    mm.destory_shared_memory()

