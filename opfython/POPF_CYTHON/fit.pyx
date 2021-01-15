
cimport cython

from  opfython.Sequential_supervised.distancepx import distancescy as distance
from libc.stdlib cimport malloc, free
from cython.parallel cimport parallel,prange
cimport openmp
from libc.stdio cimport printf   
import time
@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
cpdef fit(nodes,idx_nodes,double [:,:] features,  float Max_Cost, int n_Threads):



	cdef int size=len(features)
	cdef int lun=len(features[0])



	cdef int *P=<int *> malloc(size* sizeof(int))
	cdef int *L= <int *> malloc(size* sizeof(int))
	cdef int *U =<int *> malloc(size* sizeof(int))
	cdef double *C= <double *> malloc(size* sizeof(double))

	cdef int *RES=<int *> malloc(n_Threads* sizeof(int))
	cdef int *R1=<int *> malloc(n_Threads* sizeof(int))
	cdef int *R2=<int *> malloc(n_Threads* sizeof(int))

	#cdef int *idx=<int *> malloc(size* sizeof(int))

	cdef int i=0  
	cdef int flag=0
		
	cdef int s=0
	
	#n_Threads=1
	print(n_Threads)
	creaTagli(R1,R2,n_Threads,size)

	for i in range(size):
		U[i]=0
		if nodes[i].status==1:
			C[i]=0
			P[i]=-1

			L[i]=nodes[i].label

			if flag==0:
				s=i
				flag=1
		else:
			C[i]=Max_Cost
			P[i]=nodes[i].pred
			L[i]=0


	
	while s !=-1:
		U[s]=1
		idx_nodes.append(s)
		nodes[s].cost=C[s]

		#s=worker(features,s,size,U,L,P,C,lun,n_Threads,RES,R1,R2)
		with nogil,parallel():
			#s=worker(features,s,size,U,L,P,C,lun,n_Threads,RES,R1,R2)
			#lun=0
			for i in prange(n_Threads):
				work(features,s,size,U,L,P,C,lun,RES,R1[i],R2[i],i)
		s=minS1(RES,C,n_Threads)




	
	for i in range(size):
		nodes[i].pred = P[i]
		nodes[i].predicted_label =L[i]

	free(P)
	free(L)
	free(C)
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
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
cdef int worker(double [:,:] features,int s, int size,int *U,int *L, int *P, double *C,int lun, int n_Threads,int *RES,int *R1,int *R2):
	cdef int i
	cdef int s1
	cdef double min
	

	with nogil, parallel():
		for i in prange(n_Threads):
				work(features,s,size,U,L,P,C,lun,RES,R1[i],R2[i],i)
	

	s1=RES[0]
	min=C[s1]
	for i in range(1,n_Threads):
		if(s1==-1 or (RES[i]!=-1 and C[RES[i]]<min)):
			s1=RES[i]
			min=C[s1]
	return s1

	




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void work(double [:,:] features,int s, int size,int *U,int *L, int *P, double *C,int lun,int *RES,int r1, int r2, int i) nogil:
	#with gil:
		#print(cython.parallel.threadid(),r1,r2)
		#t1=time.time()
	cdef int t
	cdef double curr_cost
	cdef int s1=-1
	cdef double dist
	cdef int j
	
	for t in range(r1,r2):
		dist=0
		if s!=t:
			if C[t]>C[s]:

				#dist=squared_euclidean_distance(features[t],features[s],lun)
				for j in range(lun):
					dist+= (features[t][j]-features[s][j])*(features[t][j]-features[s][j])
				if C[s]>=dist:
					curr_cost=C[s]
				else:
					curr_cost=dist

				if curr_cost<C[t]:
					L[t]=L[s]
					C[t]=curr_cost
					P[t]=s

			if (s1==-1 or C[s1]>C[t]):
				if(U[t]==0):
					s1=t
	#with gil:
		#print(cython.parallel.threadid()," ",time.time()-t1)
	RES[i]=s1


	



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double squared_euclidean_distance(double [:] x, double [:]  y,int lun) nogil:
	cdef int i
	cdef double dist=0
	for i in range(lun):
		dist+= (x[i]-y[i])*(x[i]-y[i])

	return dist


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

