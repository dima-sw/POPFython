"""Supervised Optimum-Path Forest.
"""


import time

import numpy as np


import opfython.utils.constants as c

import opfython.utils.logging as log
from opfython.core import  Subgraph
from opfython.models import SupervisedOPF

from multiprocessing import JoinableQueue, Process, Queue, Array

logger = log.get_logger(__name__)


class SSupervisedPOPF(SupervisedOPF):
    """A SupervisedOPF which implements the supervised version of OPF classifier.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. Supervised Pattern Classification based on Optimum-Path Forest.
        International Journal of Imaging Systems and Technology (2009).

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

    def creaProcFit(self,target,processi,*args):

        for i in range(self._processi):
            processi.append(Process(target=target, args=(args)))
            processi[i].daemon = True
            processi[i].start()

    def creaTagli(self,tagli,parti):
        for i in range(tagli):
            if i == 0:
                r1 = 0
            else:
                r1 = i * int((self.subgraph.n_nodes / tagli))
            if i == tagli - 1:
                r2 = self.subgraph.n_nodes
            else:
                r2 = int(r1 + (self.subgraph.n_nodes / tagli))

            parti.append((r1,r2))

    def calcMin(self,risultati,tagli):
        min = risultati[0]

        j = 0
        #Vedo se ci sta almeno un minimo
        if min[0] == -1:
            for i in range(1, tagli):
                if min[0] == -1 and risultati[i][0] != -1:
                    min = risultati[i]
                    j = i
                    break

        # cerco il minimo se esiste
        for i in range(j, tagli):
            if min[1] > risultati[i][1] and risultati[i][0] != -1:
                min = risultati[i]

        return min[0]

    def _find_prototypes(self,tagli):
        """Find prototype nodes using the Minimum Spanning Tree (MST) approach.

        """

        logger.debug('Finding prototypes ...')

        start = time.time()

        P = Array('i', self.subgraph.n_nodes, lock=False)
        C = Array('f', self.subgraph.n_nodes, lock=False)
        U = Array('i', self.subgraph.n_nodes, lock=False)

        for i in range(self.subgraph.n_nodes):
            C[i]=c.FLOAT_MAX
            P[i]=c.NIL

        # Creating a Heap of size equals to number of nodes
        #h = Heap(self.subgraph.n_nodes)

        # Marking first node without any predecessor
        self.subgraph.nodes[0].pred = c.NIL
        P[0]=c.NIL

        # Adding first node to the heap
        #h.insert(0)

        p=0

        # Creating a list of prototype nodes
        prototypes = []




        percent = 0
        percOld = 1
        flagTime = True

        processi = []

        work = JoinableQueue()
        result = Queue()

        self.creaProcFit(self.protParal, processi, P, C, U, work, result)
        parti = []
        self.creaTagli(tagli, parti)





        # While the heap is not empty
        #while not h.is_empty():
        while p != -1:

            if flagTime:
                startPerc = time.time()
                flagTime = False

            # Remove a node from the heap
            #p = h.remove()

            # Gathers its cost from the heap
            self.subgraph.nodes[p].cost = C[p]
            U[p]=1
            #C[p]=

            # And also its predecessor
            #pred = self.subgraph.nodes[p].pred
            pred= P[p]


            # If the predecessor is not NIL
            if pred != c.NIL:
                # Checks if the label of current node is the same as its predecessor
                if self.subgraph.nodes[p].label != self.subgraph.nodes[pred].label:
                    # If current node is not a prototype
                    if self.subgraph.nodes[p].status != c.PROTOTYPE:
                        # Marks it as a prototype
                        self.subgraph.nodes[p].status = c.PROTOTYPE

                        # Appends current node identifier to the prototype's list
                        prototypes.append(p)

                    # If predecessor node is not a prototype
                    if self.subgraph.nodes[pred].status != c.PROTOTYPE:
                        # Marks it as a protoype
                        self.subgraph.nodes[pred].status = c.PROTOTYPE

                        # Appends predecessor node identifier to the prototype's list
                        prototypes.append(pred)

            for i in range(tagli):
                # print(tagli[i][0],tagli[i],0)
                work.put((parti[i][0], parti[i][1], p))

            #work.join() comunque devo aspettare i risultati
            # Aspetto i risultati parziali di ogni processo
            risultati = []
            for _ in range(tagli):
                risultati.append(result.get())

            # prendo il minimo
            p = self.calcMin(risultati,tagli)

            percnew = (percent / self.subgraph.n_nodes) * 100


            if (percnew > percOld):
                endPerc = time.time()
                percOld += 1
                print("Prototypes %" + str(int(percnew)))
                print("Estimeted time: " + str((endPerc - startPerc) * (100 - percnew)) + " seconds")
                flagTime = True

            percent += 1
        work.join()

        for i in range(self._processi):
            processi[i].terminate()
        for i in range(self.subgraph.n_nodes):
            self.subgraph.nodes[i].pred = P[i]

        end = time.time()
        fittime = end - start
        logger.debug('Prototypes: %s.', prototypes)
        logger.info('Prototypes found in: %s seconds.', fittime)
        return fittime

    def protParal(self,P, C, U, work, result):

        while True:
            r1,r2,p=work.get()

            s1=None

            for q in range(r1,r2):
                # Checks if the color of current node in the heap is not black
                if U[q] == 0:
                    # If `p` and `q` identifiers are different
                    if p != q:
                        # If it is supposed to use pre-computed distances
                        if self.pre_computed_distance:
                            # Gathers the arc from the distances' matrix
                            weight = self.pre_distances[self.subgraph.nodes[p].idx][self.subgraph.nodes[q].idx]

                        # If distance is supposed to be calculated
                        else:
                            # Calculates the distance
                            weight = self.distance_fn(
                                self.subgraph.nodes[p].features, self.subgraph.nodes[q].features)

                        # If current arc's cost is smaller than the path's cost
                        if weight < C[q]:
                            # Marks `q` predecessor node as `p`
                            #self.subgraph.nodes[q].pred = p
                            P[q]=p

                            # Updates the arc on the heap
                            C[q]= weight
                        if(s1 == None or C[s1] > C[q]):
                        # e se il nodo non è stato già used
                            if U[q] == 0:
                                s1 = q

                    # s1=None significa che ogni nodo del range di questo processo è già stato used
            if s1 == None:
                    s1 = -1
                    # restituisco il risultato al processo principale
            result.put((s1, C[s1]))
            work.task_done()






    def fit(self, X_train, Y_train,tagli, I_train=None):
        """Fits data in the classifier.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            I_train (np.array): Array of training indexes.

        """

        logger.info('Fitting classifier ...')

        # Initializing the timer

        # Creating a subgraph
        self.subgraph = Subgraph(X_train, Y_train, I=I_train)

        # Finding prototypes
        tt=self._find_prototypes(tagli)

        start = time.time()


        nprot=0

        P=Array('i',self.subgraph.n_nodes,lock=False)
        C=Array('f',self.subgraph.n_nodes,lock=False)
        L=Array('i', self.subgraph.n_nodes,lock=False)
        U=Array('i',self.subgraph.n_nodes,lock=False)

        flag = True  # per prendere il primo prototipo

        # For each possible node
        for i in range(self.subgraph.n_nodes):
            # Checks if node is a prototype
            if self.subgraph.nodes[i].status == c.PROTOTYPE:
                # If yes, it does not have predecessor nodes
                self.subgraph.nodes[i].pred = c.NIL
                # Its predicted label is the same as its true label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label

                U[i]=0
                C[i]=0
                P[i]=c.NIL
                L[i]=self.subgraph.nodes[i].label
                nprot+=1

                if flag:
                    #U[i]=0
                    primo = i
                    flag = False

            # If node is not a prototype
            else:
                # se non è un prototipo usato=0 cosot=MAX pred=quello che già stava nel label=nil
                U[i]=0
                C[i]=c.FLOAT_MAX
                P[i]=self.subgraph.nodes[i].pred
                L[i]=c.NIL

        """###########################################################################################################################

        """
        processi = []

        work = JoinableQueue()
        result = Queue()

        self.creaProcFit(self.train,processi,P,C,L,U, work, result)

        """"###########################################################################################################################
            
        """



        parti=[]
        self.creaTagli(tagli,parti)



        # prendo il primo prototipo
        s = primo

        percent = 0
        percOld = 1

        flagTime = True
        # quando avrò computato tutti i nodi s sarà = -1
        while s != -1:

            if flagTime:
                startPerc = time.time()
                flagTime = False

            # marchio il nodo come usato
            #partizione[s][0] = 1

            U[s]=1

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(s)

            # Gathers its cost
            #self.subgraph.nodes[s].cost = partizione[s][1]

            self.subgraph.nodes[s].cost = C[s]

            # Mando ad ogni processo s e i suoi dati contenuti in partizione[s]


            for i in range(tagli):
                #print(tagli[i][0],tagli[i],0)
                work.put((parti[i][0],parti[i][1], s))

            #work.join() comunque devo aspettare i risultati
            # Aspetto i risultati parziali di ogni processo
            risultati = []
            for _ in range(tagli):
                risultati.append(result.get())

            # prendo il minimo
            s = self.calcMin(risultati,tagli)

            


            #Tengo traccia a che punto sta, per file grossi è comodo
            percnew = (percent / self.subgraph.n_nodes) * 100

            if (percnew > percOld):
                endPerc = time.time()
                percOld += 1
                print("Training %" + str(int(percnew)))
                print("Estimeted time: " + str((endPerc - startPerc) * (100 - percnew)) + " seconds")
                flagTime = True

            percent += 1

        work.join()

        for i in range(self._processi):
            processi[i].terminate()


        # aggiorno pred e label dei nodi
        for j in range(0,self.subgraph.n_nodes):
            self.subgraph.nodes[j].pred = P[j]
            self.subgraph.nodes[j].predicted_label = L[j]

        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info('Classifier has been fitted.')
        logger.info('Training time: %s seconds.', train_time)
        return train_time+tt


    def train(self, P,C,L,U, work, result):


        while True:


            r1,r2, s = work.get()



            #U[s]=1

            # s1 dovrà essere il nodo non usato con il costo più piccolo
            s1 = None

            # lavoro solo nel mio range
            for t in range(r1, r2):

                # se non stiamo confrontando lo stesso nodo con se stesso
                if s != t:

                    # se il costo di s è più piccolo di t
                    if C[t] > C[s]:

                        if self.pre_computed_distance:
                            # Gathers the distance from the distance's matrix
                            weight = self.pre_distances[self.subgraph.nodes[s].idx][self.subgraph.nodes[t].idx]

                        # If the distance is supposed to be calculated
                        else:
                            # Calls the corresponding distance function
                            weight = self.distance_fn(
                                self.subgraph.nodes[t].features, self.subgraph.nodes[s].features)

                        # Il costo corrente sarà il massimo tra il costo dell'arco tra i due nodi (weight, l'arco in realtà non esiste) e il nodo s
                        current_cost = np.maximum(C[s], weight)

                        # If current cost is smaller than `q` node's cost
                        if current_cost < C[t]:
                            # aggiorno la label di t che sarà uguale a quella di s
                            L[t] = L[s]
                            # aggiorno il costo di t con quello corrente
                            C[t] = current_cost
                            P[t]=s



                    # se s1 non è stato ancora assegnato oppure il costo di s1>costo t1
                    if (s1 == None or C[s1] > C[t]):
                        # e se il nodo non è stato già used
                        if U[t] == 0:
                            s1 = t

            # s1=None significa che ogni nodo del range di questo processo è già stato used
            if s1 == None:
                s1 = -1
            # restituisco il risultato al processo principale
            result.put((s1, C[s1]))
            work.task_done()


