"""Distance-based metrics.
"""


cimport cython

@cython.boundscheck(False)
# this disables 'wraparound' indexing from the end of the array using negative
# indices.
@cython.wraparound(False)
cpdef float squared_euclidean_distance(double [:] x, double [:]  y):
    """Calculates the Squared Euclidean Distance.

    Args:
        x (np.array): N-dimensional array.
        y (np.array): N-dimensional array.

    Returns:
        The Squared Euclidean Distance between x and y.

    """

    # Calculates the Squared Euclidean distance for each dimension
    #dist = (x - y) ** 2
    print(x[0])

    cdef int i
    cdef float dist=0
    cdef int q=3
    for i in range(q):
        dist+= (x[i]-y[i])*(x[i]-y[i])

    return dist
