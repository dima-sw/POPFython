"""Supervised Optimum-Path Forest.

@Author: Dmytro Lysyhanych


"""

import opfython.utils.logging as log
from opfython.core import OPF


from opfython.POPF_functions.multiprocessing.predict import pred as cPred
from opfython.POPF_functions.multiprocessing.find_prot import _find_prototypes as cfind_prot
from opfython.POPF_functions.multiprocessing.fit import fit as cfit
from opfython.POPF_functions.prune import prune as cprune
from opfython.POPF_functions.learn import learn as clearn
from opfython.POPF_functions.Seq.fitting import fit as sfit
from opfython.POPF_functions.Seq.find_prototypes import _find_prototypes as sfind_prto
from opfython.POPF_functions.Seq.predict import predict as spred

from opfython.POPF_functions.numba.portNumba import _find_prototypes as numbaProt
from opfython.POPF_functions.numba.fitNumba import fit as numbaFit
from opfython.POPF_functions.numba.predictNumba import predict as numbaPred


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


    ############################
    #Sequential Python

    def seq_Find_Prototypes(self):
        sfind_prto(self)


    def seq_Fit(self,X_train, Y_train):
        sfit(self, X_train, Y_train)


    def seq_Predict(self,X_val):
        return spred(self, X_val)


    ############################

    ############################
    #Multiprocessing

    def mult_Find_Prototypes(self):
        cfind_prot(self)

    def mult_Fit(self,X_train,Y_train):
        cfit(self,X_train,Y_train)

    def mult_Predict(self,X_val):
        return cPred(self,X_val)

    ############################

    ############################
    #Numba

    def numba_Find_Prototypes(self,X_train):
        numbaProt(self, X_train)


    def numba_Fit(self,X_train,Y_train):
        self.xtrain=X_train
        numbaFit(self,X_train,Y_train)

    def numba_Predict(self,X_val):
        return numbaPred(self, X_val)
    ############################




    #Pruning
    def prune(self, X_train, Y_train, X_val, Y_val, M_loss, n_it=10, tfit="numba_Fit",tpred="numba_Predict"):
        cprune(self,X_train, Y_train, X_val, Y_val, M_loss,getattr(self,tfit),getattr(self,tpred), n_iterations=n_it)

    #Learning
    def learn(self, xt, yt, xv, yv, n_iterations=10,tdelta=0.00001,tfit="numba_Fit",tpred="numba_Predict"):
        return clearn(self, xt, yt, xv, yv,getattr(self,tfit),getattr(self,tpred), n_iterations=n_iterations,delta=tdelta)


