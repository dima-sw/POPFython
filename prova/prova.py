from multiprocessing import Array, Manager, Process
from multiprocessing.managers import BaseManager
import time
from opfython.core import Heap



"""def proc(h,i,rang):



    if i == 0:
        r1 = 0
    else:
        r1 = int(i * (rang/ 4)) + 1
    if i == 4- 1:
        r2 = rang
    else:
        r2 = int(r1 + (rang / 4))


    for ii in range(r1,r2):
        #h[ii]=ii
        h.append(ii)



class MyManager(BaseManager):
    pass
MyManager.register('Heap',Heap)

if __name__=='__main__':

    rang=50
    iniz=time.time()
    ar=[]

    for i in range(rang):
        ar.append(i)
    fin=time.time()
    print(fin-iniz)




    iniz=time.time()
    ars=[]
    ar=[]
    for i in range(4):
        ar.append(Process(target=proc, args=(ars,i,rang)))
        ar[i].deamon=True
        ar[i].start()
    for i in range(4):
        ar[i].join()
    fin=time.time()
    print(fin-iniz)"""


"""partizione=[]

for i in range(19):
    partizione.append([i,'a'])

print(partizione[1][0])

print(int(37.5))"""

s=2
p=5
t=8
print(s==None or t>p)

print(15 in range(10,20))

for i in range (0,3):
    print(i)

for i in range(4,6):
    print(i)


