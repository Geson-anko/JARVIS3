import time

x = range(1000000)
s = time.time()
for i in x:
    pass
pa = time.time() -s 

s = time.time()
for i in x:
    time.sleep(0.0)
wa = time.time() - s

d = wa - pa
print(f'passed :{pa}, time.sleep(0.0):{wa}, delta:{d}')