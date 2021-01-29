cimport cython


cpdef float squared_euclidean_distance(double [:] x, double [:]  y, int lun):
    # Calculates the Squared Euclidean distance for each dimension
    #dist = (x - y) ** 2
    
    cdef int i
    cdef float dist=0
    
    for i in range(lun):
        dist+= (x[i]-y[i])*(x[i]-y[i])

    return dist
