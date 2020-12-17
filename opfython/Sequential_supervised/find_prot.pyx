
cimport cython

from  opfython.Sequential_supervised.distancepx import distancescy as distance
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.nonecheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.

cdef void updateProt(long [:]labels,int *prot,int p,int pred):
      
      if labels[p]!=labels[pred]:

            prot[p]=1
            prot[pred]=1

    


@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)

cpdef _find_prototypes(nodes,double [:,:] features,long[:] labels,float Max_Cost):
    

    
    """cpdef int  PREDS[5000]
    cpdef float  C[5000]
    cpdef int  PROT[5000]
    cpdef int  U[5000]"""
    cdef int size=len(features)
    cdef lun=len(features[0])
    cdef int *PROT=<int *> malloc(size* sizeof(int))
    cdef int *PREDS= <int *> malloc(size* sizeof(int))
    cdef int *U =<int *> malloc(size* sizeof(int))
    cdef double *C= <double *> malloc(size* sizeof(double))


    
    cpdef int i=0  

    

    for i in range(size):
        C[i]=Max_Cost
        PREDS[i]=0
        U[i]=0
        PROT[i]=0



    cpdef int p=0

    while p!=-1:

        pr=PREDS[p]
        U[p]=1

        if pr!=0:
            updateProt(labels,PROT,p,pr)


        p=work(features,PREDS,C,U,p,size,lun)

    
    for i in range(size):
                nodes[i].pred = PREDS[i]
                nodes[i].status=PROT[i]

    free(PREDS)
    free(C)
    free(PROT)
    free(U)
    #free(PROT)



@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)

cdef int work(double [:,:] features, int *PREDS, double *C, int *U,int p,int size,int lun):
    
    
    cdef int s1=-1

    cdef int q=0



    for q in range(size):
        if U[q] == 0:

            if p!=q:

                dist=squared_euclidean_distance(features[p],features[q],lun)
                if dist<=C[q]:
                    PREDS[q]=p

                    C[q]=dist

                if (s1==-1 or C[s1]>C[q]):
                    if U[q]==0:
                        s1=q


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


