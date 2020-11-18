"""Supervised Parallel Optimum-Path Forest.
"""

import opfython.utils.logging as log
from opfython.core import OPF


from opfython.POPF_functions.predict import pred as cPred
from opfython.POPF_functions.find_prot import _find_prototypes as find_prot
from opfython.POPF_functions.fit import fit as cfit
from opfython.POPF_functions.prune import prune as cprune
from opfython.POPF_functions.learn import learn as clearn
import math
logger = log.get_logger(__name__)


class SSupervisedPOPF(OPF):
    """
    """

    def __init__(self, processi=4, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.
        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.
        """

        logger.info('Overriding class: OPF -> SupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SSupervisedPOPF, self).__init__(distance, pre_computed_distance)

        self._processi = processi

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


    def _find_prototypes(self,tagli):
        find_prot(self,tagli)


    def fit(self,X_train, Y_train,tagli):
        cfit(self,X_train,Y_train,tagli)


    def prune(self, X_train, Y_train, X_val, Y_val, tagli, M_loss, n_it=10):
        cprune(self,X_train, Y_train, X_val, Y_val, tagli, M_loss, n_iterations=n_it)


    def learn(self, xt, yt, xv, yv, tagli, n_iterations=10):
        return clearn(self, xt, yt, xv, yv, tagli, n_iterations=n_iterations)

    def pred(self, X_val, tagli):
        return cPred(self, X_val, tagli)