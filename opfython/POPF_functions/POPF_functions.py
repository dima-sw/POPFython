from multiprocessing import Process, Queue, JoinableQueue

def creaProcFit(target, processi,nproc, *args):
    for i in range(nproc):
        processi.append(Process(target=target, args=(args)))
        processi[i].daemon = True
        processi[i].start()


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

def calcMin(result):
    r=result.get()
    s = r[0]
    min=r[1]
    while not result.empty():
        r=result.get()
        if (min>r[1] and r[0]!=-1) or (s==-1 and r[0]!=-1):
            s=r[0]
            min=r[1]
    return s