
cimport cython

from  opfython.Sequential_supervised.distancepx import distancescy as distance
from libc.stdlib cimport malloc, free


    
@cython.boundscheck(False)

@cython.wraparound(False)
cpdef fit(nodes,idx_nodes,double [:,:] features,  float Max_Cost):




    """
        size= numero di nodi
        P=predecessori
        L=labels
        U=used
        C=costo di ogni nodo
        idx=Array di nodi ordinato in termini di costo
        lun=numero di feature di ogni nodo

    """
	cdef int size=len(features)

	cdef int *P=<int *> malloc(size* sizeof(int))
	cdef int *L= <int *> malloc(size* sizeof(int))
	cdef int *U =<int *> malloc(size* sizeof(int))
	cdef double *C= <double *> malloc(size* sizeof(double))
	#cdef int *idx=<int *> malloc(size* sizeof(int))

	cdef int i=0  
	cdef int flag=0
	cdef lun=len(features[0])
	cdef int s=0


    #Inizializzo gli array in base se sono prototipi o meno
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


    #TRAINING, quando sara' finito s sara' -1, il primo s sara' un prototipo siccome ha costo 0
	while s !=-1:

	    #Setto il nodo come usato, lo aggiungo alla lista ordinata e aggiorno il costo del nodo reale
		U[s]=1
		idx_nodes.append(s)
		nodes[s].cost=C[s]

        #mi prendo il prossimo s
		s=work(features,s,size,U,L,P,C,lun)




	
	for i in range(size):
		nodes[i].pred = P[i]
		nodes[i].predicted_label =L[i]

	free(P)
	free(L)
	free(C)
	free(U)
    #free(PROT)


@cython.boundscheck(False)
@cython.wraparound(False)
#Funzione che fa il training e restituisce il nodo col costo minore
cdef int work(double [:,:] features,int s, int size,int *U,int *L, int *P, double *C,int lun):
	cdef int t
	cdef double curr_cost
	cdef int s1=-1

    #itero per ogni nodo
	for t in range(size):
	    #se s e t non sono uguali
		if s!=t:
		    #Se il costo di t e' maggiore del costo di s allora vedo se s possa conquistare t
			if C[t]>C[s]:

				dist=squared_euclidean_distance(features[t],features[s],lun)

				if C[s]>=dist:
					curr_cost=C[s]
				else:
					curr_cost=dist

				if curr_cost<C[t]:
					L[t]=L[s]
					C[t]=curr_cost
					P[t]=s
			#Vedo se t possa essere il prossimo nodo ad uscire
			if (s1==-1 or C[s1]>C[t]):
				if(U[t]==0):
					s1=t
	return s1


	



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


