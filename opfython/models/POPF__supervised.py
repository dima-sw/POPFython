"""Supervised Parallel Optimum-Path Forest.
"""

import opfython.utils.logging as log
from opfython.core import OPF


from opfython.POPF_functions.predict import pred as cPred
from opfython.POPF_functions.find_prot import _find_prototypes as cfind_prot
from opfython.POPF_functions.fit import fit as cfit
from opfython.POPF_functions.prune import prune as cprune
from opfython.POPF_functions.learn import learn as clearn
from opfython.POPF_functions.Seq.fitting import fit as sfit
from opfython.POPF_functions.Seq.find_prototypes import _find_prototypes as sfind_prto
from opfython.POPF_functions.Seq.predict import predict as spred

from opfython.POPF_functions.numba.portNumba import _find_prototypes as numbaProt
from opfython.POPF_functions.numba.fitNumba import fit as numbaFit
from opfython.POPF_functions.numba.predictNumba import predict as numbaPred
import time


import math
logger = log.get_logger(__name__)


class SSupervisedPOPF(OPF):
    """
    """

    def __init__(self, processi=4,tagli=10, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.
        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.
        """

        logger.info('Overriding class: OPF -> SupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SSupervisedPOPF, self).__init__(distance, pre_computed_distance)

        self._processi = processi
        self._tagli=tagli
        logger.info('Class overrided.')

    def calcWeight(self, s, t):
        if self.pre_computed_distance:
            # Gathers the distance from the distance's matrix
            weight = self.pre_distances[self.subgraph.nodes[s].idx][self.subgraph.nodes[t].idx]

        # If the distance is supposed to be calculated
        else:
            # Calls the corresponding distance function
            weight = self.distance_fn(
                self.subgraph.nodes[t].features, self.subgraph.nodes[s].features)
        return weight

    #Find prototypes using MST approach
    def _find_prototypes(self):
        #if self.subgraph.n_nodes<=500:
        #sfind_prto(self)
        #else:
        #cfind_prot(self)
        t1=time.time()
        numbaProt(self,self.xtrain)
        print("findProt: ",time.time()-t1)

    #Training
    def fit(self,X_train, Y_train):
        #if len(X_train)<=500:
        #sfit(self,X_train,Y_train)
        #else:
        self.xtrain=X_train
        #cfit(self,X_train,Y_train)
        numbaFit(self,X_train,Y_train)

    #Pruning
    def prune(self, X_train, Y_train, X_val, Y_val, M_loss, n_it=10):
        cprune(self,X_train, Y_train, X_val, Y_val, M_loss, n_iterations=n_it)

    #Learning
    def learn(self, xt, yt, xv, yv, n_iterations=10):
        return clearn(self, xt, yt, xv, yv, n_iterations=n_iterations)

    #Predict
    def pred(self, X_val):
        #if(len(X_val)<=500):
        #return spred(self,X_val)
        #return cPred(self, X_val)

        return numbaPred(self,X_val)

    def predM(self,X_val):
        return cPred(self, X_val)
