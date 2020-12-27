from multiprocessing import Process, Queue, JoinableQueue
from threading import Thread
import numpy as np

def creaProcFitf(target, processi,nproc, *args):
    for i in range(nproc):

        processi.append(Process(target=target, args=(*args,i)))
        processi[i].daemon = True
        processi[i].start()


def creaProcFit(target, processi,nproc, *args):
    for i in range(nproc):
        processi.append(Process(target=target, args=(args)))
        processi[i].daemon = True
        processi[i].start()

def creaThreadFit(target, processi,nproc, *args):
    for i in range(nproc):
        processi.append(Thread(target=target, args=(args)))
        processi[len(processi)-1].setDaemon(True)
        processi[len(processi)-1].start()



def creaTagli(tagli, parti, n):
    for i in range(tagli):
        if i == 0:
            r1 = 0
        else:
            r1 = i * int((n / tagli))
        if i == tagli - 1:
            r2 = n
        else:
            r2 = int(r1 + (n / tagli))
        parti.append((r1, r2))


def creaTagliNP(tagli,n):
    tt = np.full((tagli, 2), 0, dtype=np.int)

    for i in range(tagli):
        if i == 0:
            r1 = 0
        else:
            r1 = i * int((n / tagli))
        if i == tagli - 1:
            r2 = n
        else:
            r2 = int(r1 + (n / tagli))
        tt[i][0]=r1
        tt[i][1]=r2
    return tt


def calcMin(result):


    r=result.get()
    s = r[0]
    min=r[1]
    while not result.empty():
        r=result.get()
        #if l == -1 or (res[i] != -1 and C[res[i]] < C[l]):
        if s==-1 or (r[0]!=-1 and min>r[1]):
        #if (min>r[1] or s==-1) and r[0]!=-1:
            s=r[0]
            min=r[1]
    return s




