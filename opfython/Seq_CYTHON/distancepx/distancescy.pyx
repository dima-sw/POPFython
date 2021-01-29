cimport cython


cpdef float squared_euclidean_distance(double [:] x, double [:]  y):
    # Calculates the Squared Euclidean distance for each dimension
    #dist = (x - y) ** 2
    
    cdef int i
    cdef float dist=0
    cdef int lun=len(x)
    for i in range(lun):
        dist+= (x[i]-y[i])*(x[i]-y[i])

    return dist
