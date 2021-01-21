import time

import opfython.utils.constants as c
from opfython.POPF_functions.POPF_functions import creaProcFit,creaTagliNP,calcMin,creaThreadFit
import numpy as np
from numba import njit,prange, set_num_threads


#Aggiornamento dei prototipi
def updateProt(opf, prototypes, p, pred):
    # Checks if the label of current node is the same as its predecessor
    if opf.subgraph.nodes[p].label != opf.subgraph.nodes[pred].label:
        # If current node is not a prototype
        if opf.subgraph.nodes[p].status != c.PROTOTYPE:
            # Marks it as a prototype
            opf.subgraph.nodes[p].status = c.PROTOTYPE
            # Appends current node identifier to the prototype's list
            prototypes.append(p)

        # If predecessor node is not a prototype
        if opf.subgraph.nodes[pred].status != c.PROTOTYPE:
            # Marks it as a protoype
            opf.subgraph.nodes[pred].status = c.PROTOTYPE

            # Appends predecessor node identifier to the prototype's list
            prototypes.append(pred)


def _find_prototypes(opf,xtrain):

    start=time.time()

    #Setto il numero dei thread
    set_num_threads(opf._processi)

    """Questi 3 array li useremo durante la concorrenza
            P-> array dei pred
            C-> array dei costi
            U-> array degli used per vedere se gia' abbiamo usato il nodo
    """
    P=np.full(opf.subgraph.n_nodes,c.NIL,dtype=np.int)
    C=np.full(opf.subgraph.n_nodes,c.FLOAT_MAX,dtype=np.float)
    U=np.full(opf.subgraph.n_nodes,0,dtype=np.int)



    #Lista dei prototipi
    prototypes = []
    #Il primo nodo
    p=0


    #Creo le slice
    slices=creaTagliNP(opf._processi,opf.subgraph.n_nodes)
    #Array dei risultati
    res=np.full(opf._processi,-1,dtype=np.int)

    #Parte mst da p
    while p!=-1:
        #aggiorno il costo del nodo p del grafo
        opf.subgraph.nodes[p].cost=C[p]
        pred=P[p]
        U[p]=1

        #Se pred di p non e' nil allora vedo se possono essere prototipi
        if pred != c.NIL:
            updateProt(opf,prototypes,p,pred)
        #Parte il work sul nodo p
        worker(P,C,U,p,xtrain,slices,res,opf._processi)
        #Calcolo il prossimo p
        p=calcMin(C,res)


    #Aggiorno il grafo
    for i in range(opf.subgraph.n_nodes):
        opf.subgraph.nodes[i].pred = (int)(P[i])

    print("Protypes found in: ", time.time()-start)


#Questa funzione calcola il minimo s1 in termine di costo
def calcMin(C,res):
    l=res[0]
    for i in range(1,len(res)):
        if l==-1 or (res[i]!=-1 and C[res[i]]<C[l]):
            l=res[i]
    return l



@njit(fastmath=True,nogil=True,parallel=True,cache=True)
def worker(P,C,U,p,xtrain,slice,res,n_threads):
    #Suddivido il lavoro in N threads ed a ciascuno do una slice sulla quale lavorare
    for i in prange(n_threads):
        work(P,C,U,p,xtrain,res,slice[i][0],slice[i][1],i)



@njit(fastmath=True,nogil=True,cache=True)
def work(P,C,U,p,xtrain,res,r1,r2,i):
    s1 = -1
    for q in range(r1,r2):
        if U[q] == 0:
            if p != q:
                weight = calcWeight(xtrain[p], xtrain[q])
                #Se la distanza e' più piccola del costo di q allora aggiorno P di q e C di q
                if weight < C[q]:
                    P[q] = p
                    C[q] = weight
                #Vedo se q possa essere il prossimo nodo ad uscire
                if (s1 == -1 or C[s1] > C[q]):
                    if U[q] == 0:
                        s1 = q
    res[i]=s1



#Calcolo la distanza euclidiana tra due nodi
@njit(fastmath=True,nogil=True,cache=True)
def calcWeight(x,y):
    dist = 0

    for i in range(len(x)):
        dist += (x[i] - y[i]) * (x[i] - y[i])

    return dist