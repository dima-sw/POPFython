
cimport cython

from  opfython.Sequential_supervised.distancepx import distancescy as distance
from libc.stdlib cimport malloc, free



    
@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
cpdef predict(double [:] cost,long [:] predicted_label,long [:] ordered_nodes,double [:,:] features,double [:,:] val_feat,long [:]P2,long [:] L2):
	cdef int *P=<int *> malloc(len(val_feat)* sizeof(int))
	cdef int p_indx=0
	cdef int l_indx=0
	cdef int f_len=len(features[0])
	cdef int nodes_len=len(features)
	cdef int conqueror
	cdef int j

	cdef int i

	cdef double dist
	cdef double min_cost
	cdef double tmp_cost
	cdef int current_label
	for i in range(len(val_feat)):
		conqueror=-1
		j=0
		k=ordered_nodes[j]


		dist= squared_euclidean_distance(features[k],val_feat[i],f_len)
		if cost[k]>dist:
			min_cost=cost[k]
		else:
			min_cost=dist

		current_label=predicted_label[k]
		while j<(nodes_len-1) and min_cost > cost[ordered_nodes[j+1]]:
			l=ordered_nodes[j+1]


			dist=squared_euclidean_distance(features[l],val_feat[i],f_len)
			
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
		if conqueror> -1:
			P2[l_indx]=conqueror
			l_indx+=1
		p_indx+=1

	#i=0
	#for i in range(len(val_feat)):
		#L2.append(L[i])

	free(P)
	
	




@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
cdef float squared_euclidean_distance(double [:] x, double [:]  y,int lun):
	cdef int i
	cdef float dist=0
	for i in range(lun):
		dist+= (x[i]-y[i])*(x[i]-y[i])

	return dist


