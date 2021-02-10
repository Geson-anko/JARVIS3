from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
import numpy as np


def change(array,pubmem):
    array = np.ndarray(shape=array.shape,dtype=array.dtype,buffer=pubmem.buf)
    print('change:',array.data)
    print('change:',array)
    array[:] = 1
    print('change:',array)

if __name__ == '__main__':
    dummy = np.random.randn(3)
    pubmem = shared_memory.SharedMemory(create=True,size=dummy.nbytes)
    array = np.ndarray(shape=dummy.shape,dtype=dummy.dtype,buffer=pubmem.buf)
    array[:] = 0
    print('pubmem:',pubmem.buf,pubmem.name)
    print('first',array.data)
    print('first',array)
    
    with ProcessPoolExecutor() as p:
        p.submit(change,array,pubmem)
    
    print('finished:',array)
    pubmem.close()
    pubmem.unlink()