"""
Parallel Find Prototypes (MST Approach)

@Author: Dmytro Lysyhanych

"""
cimport cython

from  opfython.Sequential_supervised.distancepx import distancescy as distance
from libc.stdlib cimport malloc, free
from cython.parallel cimport parallel,prange
cimport openmp
from libc.stdio cimport printf   
import time


cdef void updateProt(long [:]labels,int *prot,int p,int pred):
      if labels[p]!=labels[pred]:
            prot[p]=1
            prot[pred]=1





@cython.boundscheck(False)
@cython.wraparound(False)
cpdef _find_prototypes(nodes,double [:,:] features,long[:] labels,float Max_Cost,int n_Threads):

    """
        nodes: nodi del grafo
        features: matrice  di features per il training, ogni riga rapresenta un array di features per ogni nodo
        labels: le label per ogni ogni nodo
        Max_Cost: costante che definisce il numero float più grande possibile (simulare l'infinito)
        n_Threads: numero di Thread

    """

    #size: numero dei nodi
    cdef int size=len(features)
    #lun= numero di features per ogni nodo
    cdef int lun=len(features[0])



    """
        PROT:array di prototipi, 0=non prototipo 1=prototipo
        L:array di labels
        U:Quando un nodo i entra nel MST U[i]=1
        C:Array dei costi dei nodi
        PREDS:Array dei predecessori

        RES:Array dei minimi dei Thread
        R1=Array di inizio del range sul quale opera ogni Thread
        R2:Array del fine del range sul quale opera ogni Thread

    """
    cdef int *PROT=<int *> malloc(size* sizeof(int))
    cdef int *L= <int *> malloc(size* sizeof(int))
    cdef int *U =<int *> malloc(size* sizeof(int))
    cdef int *PREDS= <int *> malloc(size* sizeof(int))
    cdef double *C= <double *> malloc(size* sizeof(double))

    cdef int *RES=<int *> malloc(n_Threads* sizeof(int))
    cdef int *R1=<int *> malloc(n_Threads* sizeof(int))
    cdef int *R2=<int *> malloc(n_Threads* sizeof(int))



    cdef int i=0  

    #Predecessore del nodo corrente
    cdef int pr

    #Inizializzo gli array
    for i in range(size):
        C[i]=Max_Cost
        PREDS[i]=0
        U[i]=0
        PROT[i]=0

    #Primo nodo
    cdef int p=0

    #Creo le slice
    creaTagli(R1,R2,n_Threads,size)


    #Inizia il find Prototype
    while p !=-1:
        U[p]=1
        pr=PREDS[p]

        #Se il nodo ha un predecessore, chiamiamo updatedProt che controlla se hanno la stessa label o meno, in caso affermativo diventano prototipi
        if pr!=0:
            updateProt(labels,PROT,p,pr)

        #SEZIONE SENZA IL GIL
        with nogil,parallel():

            #Suddivido il lavoro per n_Threads
            for i in prange(n_Threads):
                #Passo i features, l'array dei predecessori, dei costi, used, il nodo p, il numero di features per ogni nodo, l'array dei risultati
                #Il range inizio e il range di fine ed infine l'indice del Thread che serve per scrivere il risultato nella sua locazione di memoria
                work(features,PREDS,C,U,p,lun,RES,R1[i],R2[i],i)
        #Calcolo il minimo dei minimi
        p=minS1(RES,C,n_Threads)




    #Aggiorno il grafo
    for i in range(size):
                nodes[i].pred = PREDS[i]
                nodes[i].status=PROT[i]

    free(PREDS)
    free(C)
    free(PROT)
    free(U)
    free(RES)
    free(R1)
    free(R2)
    #free(PROT)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int minS1(int *RES,double *C,n_Threads): #Calcolo il minimo dei minimi
    s1=RES[0]
    min=C[s1]
    for i in range(1,n_Threads):
        if(s1==-1 or (RES[i]!=-1 and C[RES[i]]<min)):
            s1=RES[i]
            min=C[s1]
    return s1




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void work(double [:,:] features,int *PREDS, double *C, int *U,int p,int lun,int *RES,int r1, int r2, int i) nogil:

    cdef int j
    cdef int s1=-1
    cdef int q
    cdef double dist



    for q in range(r1,r2):
        if U[q]==0:

            dist=0
            if p!=q:

                    for j in range(lun):
                        dist+= (features[q][j]-features[p][j])*(features[q][j]-features[p][j])
                    if dist<=C[q]:
                        PREDS[q]=p

                        C[q]=dist

                    if (s1==-1 or C[s1]>C[q]):
                        if U[q]==0:
                            s1=q

    RES[i]=s1


    


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void creaTagli(int *R1,int *R2,int nTagli,int n_nodes):
    cdef int i
    for i in range(nTagli):
        if i==0:
            R1[i]=0
        else:
            R1[i]= i * int((n_nodes/nTagli))
        if i==nTagli-1:
            R2[i]=n_nodes
        else:
            R2[i]=int(R1[i]+ (n_nodes/nTagli))

