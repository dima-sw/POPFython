import time
import numpy as np
from numba import njit,prange, set_num_threads
from opfython.POPF_functions.POPF_functions import  creaTagliNP
def predict(opf,X_val,I_val=None):
    set_num_threads(opf._processi)

    t1 = time.time()
    xt=opf.xtrain
    xv=X_val


    tt=creaTagliNP(opf._processi,len(xv))



    p_labels=np.asarray([n.label for n in opf.subgraph.nodes])
    p_cost=np.asarray([n.cost for n in opf.subgraph.nodes])
    P2=np.zeros(len(X_val),dtype=np.int)
    L2=np.zeros(len(X_val),dtype=np.int)
    ordered_nodes=np.asarray(opf.subgraph.idx_nodes)

    pp(xt,xv,p_labels,p_cost,P2,L2,ordered_nodes,tt,opf._processi)

    print("Classification in: ", time.time() - t1)

    return L2,P2

@njit(fastmath=True,nogil=True,parallel=True,cache=True)
def pp(xt,xv,pl,pc,P2,L2,ord_n,slice,t):
    for i in prange(t):
        worker(xt,xv,pl,pc,P2,L2,ord_n,slice[i][0],slice[i][1])



@njit(fastmath=True,nogil=True,cache=True)
def worker(xt,xv,pl,pc,P2,L2,ord_n,r1,r2):

    for i in range(r1,r2):
        j = 0
        k = ord_n[j]
        conqueror = k

        dist = calcWeight(xt[k], xv[i])
        min_cost = np.maximum(pc[k], dist)

        current_label = pl[k]

        while j < (len(xt) - 1) and min_cost > pc[ord_n[j + 1]]:
            l = ord_n[j + 1]

            dist = calcWeight(xt[l], xv[i])
            tmp_cost = np.maximum(pc[l], dist)

            if tmp_cost < min_cost:
                min_cost = tmp_cost
                conqueror = l
                current_label = pl[l]
            j += 1
            # k=l
        L2[i] = current_label
        P2[i] = conqueror



@njit(fastmath=True,nogil=True,cache=True)
def calcWeight(x,y):
    dist = 0

    for i in range(len(x)):
        dist += (x[i] - y[i]) * (x[i] - y[i])

    return dist