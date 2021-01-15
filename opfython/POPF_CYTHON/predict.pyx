
cimport cython

from  opfython.Sequential_supervised.distancepx import distancescy as distance
from libc.stdlib cimport malloc, free

from cython.parallel cimport parallel,prange

    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef predict(double [:] cost,long [:] predicted_label,long [:] ordered_nodes,double [:,:] features,double [:,:] val_feat,long [:]P2,long [:] L2,int n_Threads):
	#cdef int *P=<int *> malloc(len(val_feat)* sizeof(int))
	
	cdef int f_len=len(features[0])
	cdef int nodes_len=len(features)
	cdef int *R1=<int *> malloc(n_Threads* sizeof(int))
	cdef int *R2=<int *> malloc(n_Threads* sizeof(int))

	creaTagli(R1,R2,n_Threads,len(val_feat))
	cdef int i
	with nogil,parallel():
			for i in prange(n_Threads):
				work(cost,predicted_label,ordered_nodes,features,val_feat,P2,L2,f_len,nodes_len,R1[i],R2[i])
	
	

	free(R1)
	free(R2)


@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)	
cdef void ww(double [:] cost,long [:] predicted_label,long [:] ordered_nodes,double [:,:] features,double [:,:] val_feat,long [:]P2,long [:] L2,int f_len,int nodes_len,int r1,int r2,int i) nogil:
	pass




@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)	
cdef void work(double [:] cost,long [:] predicted_label,long [:] ordered_nodes,double [:,:] features,double [:,:] val_feat,long [:]P2,long [:] L2,int f_len,int nodes_len,int r1,int r2) nogil:
	
	cdef int j,i,k,l,z

	
	cdef int p_indx=r1
	cdef int conqueror
	cdef double dist
	cdef double min_cost
	cdef double tmp_cost
	cdef int current_label
	
	for i in range(r1,r2):
		conqueror=0
		j=0
		k=ordered_nodes[j]
		dist=0

		for z in range(f_len):
					dist+= (features[k][z]-val_feat[i][z])*(features[k][z]-val_feat[i][z])
		if cost[k]>dist:
			min_cost=cost[k]
		else:
			min_cost=dist

		current_label=predicted_label[k]
		while j<(nodes_len-1) and min_cost > cost[ordered_nodes[j+1]]:
			l=ordered_nodes[j+1]


			dist=0
			for z in range(f_len):
					dist+= (features[l][z]-val_feat[i][z])*(features[l][z]-val_feat[i][z])
			if(cost[l]>dist):
				tmp_cost=cost[l]
			else:
				tmp_cost=dist

			if tmp_cost<min_cost:
				min_cost=tmp_cost
				conqueror=l
				current_label=predicted_label[l]

			j+=1
			k=l

		L2[p_indx]=current_label
		
		P2[p_indx]=conqueror
		p_indx+=1





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