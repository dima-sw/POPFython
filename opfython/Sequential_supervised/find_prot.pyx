
cimport cython

from  opfython.Sequential_supervised.distancepx import distancescy as distance
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.nonecheck(False)
#Faccio update dei nodi se non hanno la stessa label
cdef void updateProt(long [:]labels,int *prot,int p,int pred):
      
      if labels[p]!=labels[pred]:

            prot[p]=1
            prot[pred]=1

    


@cython.boundscheck(False)
@cython.wraparound(False)

cpdef _find_prototypes(nodes,double [:,:] features,long[:] labels,float Max_Cost):
    

    
    """cpdef int  PREDS[5000]
    cpdef float  C[5000]
    cpdef int  PROT[5000]
    cpdef int  U[5000]"""

    """
        size: numero di nodi
        lun: numedo di features per ogni nodo
        PROT: se un nodo e' prototipo oppure no
        PREDS: i predecessori di ogni nodo
        U: se il nodo e' gia' entrato nel MST
        C: costo di ogni nodo

    """
    cdef int size=len(features)
    cdef lun=len(features[0])
    cdef int *PROT=<int *> malloc(size* sizeof(int))
    cdef int *PREDS= <int *> malloc(size* sizeof(int))
    cdef int *U =<int *> malloc(size* sizeof(int))
    cdef double *C= <double *> malloc(size* sizeof(double))


    
    cpdef int i=0  

    #Inizializzo tutti i nodi con costo massimo, nessun predecessore, non entrato, non prototipo
    for i in range(size):
        C[i]=Max_Cost
        PREDS[i]=0
        U[i]=0
        PROT[i]=0



    cpdef int p=0

    #Quando saranno entrati tutti nel MST, p sara' -1
    while p!=-1:

        pr=PREDS[p]
        U[p]=1

        #Se il predecessore esiste vedo se hanno la stessa label se no allora sono prototipi
        if pr!=0:
            updateProt(labels,PROT,p,pr)

        #Work mi dara' il prossimo nodo che entra nel MST
        p=work(features,PREDS,C,U,p,size,lun)

    #Aggiorno il nodi reali
    for i in range(size):
                nodes[i].pred = PREDS[i]
                nodes[i].status=PROT[i]
    #Libero la memoria
    free(PREDS)
    free(C)
    free(PROT)
    free(U)
    #free(PROT)



@cython.boundscheck(False)
@cython.wraparound(False)

#La funzione che trova il prossimo nodo che entra nel MST
cdef int work(double [:,:] features, int *PREDS, double *C, int *U,int p,int size,int lun):
    
    #Sara' il prossimo nodo ad entrare
    cdef int s1=-1

    cdef int q=0

    #Itero per ogni nodo
    for q in range(size):

        #Il nodo non deve gia' stare nel MST
        if U[q] == 0:
            #E deve essere diverso da quello corrente
            if p!=q:
                #Calcolo la distanza tra i due nodi
                dist=squared_euclidean_distance(features[p],features[q],lun)

                #se la distanza e' minore del costo attuale del nodo allora aggiorno il predecessore e il costo
                if dist<=C[q]:
                    PREDS[q]=p

                    C[q]=dist
                #vedo se questo possa essere il prossimo nodo ad uscire
                if (s1==-1 or C[s1]>C[q]):
                    if U[q]==0:
                        s1=q

    #Ritorno il prossimo nodo che esce
    return s1




cdef float squared_euclidean_distance(double [:] x, double [:]  y,int lun):
    """Calculates the Squared Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Euclidean Distance between x and y.

    """

    # Calculates the Squared Euclidean distance for each dimension
    #dist = (x - y) ** 2
    

    cdef int i
    cdef float dist=0
    for i in range(lun):
        dist+= (x[i]-y[i])*(x[i]-y[i])

    return dist


