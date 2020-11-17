from multiprocessing import Array, Manager, Process, JoinableQueue, Queue
from multiprocessing.managers import BaseManager
import time
from opfython.core import Heap

#from PIL import Image, ImageEnhance, ImageFilter,ImageOps
from numpy import asarray
import copy

"""def make_square(im, min_size=256, fill_color=(0, 0, 0)):
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


i=Image.open('/home/dima/Desktop/testing/calo.jpeg')
p=make_square(i).convert('L')
print(p.size)
p.show()

print(asarray(p).min(), asarray(p).max())"""

def f(s):
    #s=[3,4,5]

    t=copy.deepcopy(s)

    t[0][0]=22


    #s[:]=t[:]
    print(s[:]==t[:])
    return t



s=[[3,4,5],[3,4,5]]
t=f(s)
print(s[:]==t[:])

"""def proc(queue,i):

    while True:

        q=queue.get()

        print(q)

        queue.task_done()



queue=JoinableQueue()

p=Process(target=proc, args=(queue,10))
p.daemon=True
p.start()


queue.put(10)

queue.put(5)

queue.join()

p.terminate()

time.sleep(2)
print(p.is_alive())"""



















"""def f(i,t, ar):
    ar[i]=t
    print(ar)




a=Array('f',30)
x=[]
for i in range(8):
    x.append(Process(target=f, args=(i,(i,i),a)))
    x[i].daemon=True
    x[i].start()

for i in range(8):
    x[i].join()
    print(a[i])"""





"""path="/home/dima/Desktop/testing/gall.jpg"
path2="/home/dima/Desktop/testing/ch.jpg"

im=Image.open(path)
ch=Image.open(path2)
w, h= im.size

#m.crop((0,h/2,w/2,h)).show()

ImageEnhance.Brightness(im).enhance(1.5).convert('L').show()"""

#im.filter(ImageFilter.GaussianBlur(radius=5)).show()
















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

print(int(37.5))

s=2
p=5
t=8
print(s==None or t>p)

print(15 in range(10,20))

for i in range (0,3):
    print(i)

for i in range(4,6):
    print(i)
"""





