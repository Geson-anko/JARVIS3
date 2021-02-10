from concurrent.futures import ThreadPoolExecutor
import time

def wait(x1,x2):
    time.sleep(3)
    print(x1,x2)

insuu = [(1,2),(3,4),(5,6)]

if __name__ == '__main__':
    with ThreadPoolExecutor() as p:
        p.map(wait,insuu) # <- failed