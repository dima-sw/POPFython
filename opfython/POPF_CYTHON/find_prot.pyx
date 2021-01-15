
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
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
cpdef _find_prototypes(nodes,double [:,:] features,long[:] labels,float Max_Cost,int n_Threads):



    cdef int size=len(features)
    cdef int lun=len(features[0])




    cdef int *PROT=<int *> malloc(size* sizeof(int))
    cdef int *L= <int *> malloc(size* sizeof(int))
    cdef int *U =<int *> malloc(size* sizeof(int))
    cdef int *PREDS= <int *> malloc(size* sizeof(int))
    cdef double *C= <double *> malloc(size* sizeof(double))

    cdef int *RES=<int *> malloc(n_Threads* sizeof(int))
    cdef int *R1=<int *> malloc(n_Threads* sizeof(int))
    cdef int *R2=<int *> malloc(n_Threads* sizeof(int))

    #cdef int *idx=<int *> malloc(size* sizeof(int))

    cdef int i=0  
    
    cdef int pr
    for i in range(size):
        C[i]=Max_Cost
        PREDS[i]=0
        U[i]=0
        PROT[i]=0


    cdef int p=0

    creaTagli(R1,R2,n_Threads,size)


    
    while p !=-1:
        U[p]=1
        pr=PREDS[p]

        if pr!=0:
            updateProt(labels,PROT,p,pr)

        #s=worker(features,s,size,U,L,P,C,lun,n_Threads,RES,R1,R2)
        with nogil,parallel():
            #s=worker(features,s,size,U,L,P,C,lun,n_Threads,RES,R1,R2)
            #lun=0
            for i in prange(n_Threads):
                work(features,PREDS,C,U,p,lun,RES,R1[i],R2[i],i)
        p=minS1(RES,C,n_Threads)




    
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
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
cdef int minS1(int *RES,double *C,n_Threads):
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
    #with gil:
        #print(cython.parallel.threadid(),r1,r2)
        #t1=time.time()
    cdef int j
    cdef int s1=-1
    cdef int q
    cdef double dist



    for q in range(r1,r2):
        if U[q]==0:

            dist=0
            if p!=q:

                    #dist=squared_euclidean_distance(features[t],features[s],lun)
                    for j in range(lun):
                        dist+= (features[q][j]-features[p][j])*(features[q][j]-features[p][j])
                    if dist<=C[q]:
                        PREDS[q]=p

                        C[q]=dist

                    if (s1==-1 or C[s1]>C[q]):
                        if U[q]==0:
                            s1=q
    #with gil:
        #print(cython.parallel.threadid()," ",time.time()-t1)
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

