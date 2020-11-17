"""Supervised Optimum-Path Forest.
"""


import time

import numpy as np
import copy
import opfython.math.random as r
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as log
from opfython.core import  Subgraph,OPF
from opfython.models import SupervisedOPF
import opfython.math.general as g
from multiprocessing import JoinableQueue, Process, Queue, Array

import math
logger = log.get_logger(__name__)


class SSupervisedPOPF(OPF):
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

    def creaTagli(self,tagli,parti,n):
        for i in range(tagli):
            if i == 0:
                r1 = 0
            else:
                r1 = i * int((n / tagli))
            if i == tagli - 1:
                r2 = n
            else:
                r2 = int(r1 + (n / tagli))
            parti.append((r1,r2))

    def calcWeight(self,s,t):
        if self.pre_computed_distance:
            # Gathers the distance from the distance's matrix
            weight = self.pre_distances[self.subgraph.nodes[s].idx][self.subgraph.nodes[t].idx]

        # If the distance is supposed to be calculated
        else:
            # Calls the corresponding distance function
            weight = self.distance_fn(
                self.subgraph.nodes[t].features, self.subgraph.nodes[s].features)
        return weight

    """Calcolo il minimo dei risultati"""
    def calcMin(self,result):
        r=result.get()
        s = r[0]
        min=r[1]
        while not result.empty():
            r=result.get()
            if (min>r[1] and r[0]!=-1) or (s==-1 and r[0]!=-1):
                s=r[0]
                min=r[1]
        return s


    def updateProt(self,prototypes,p,pred):
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

    def _find_prototypes(self,tagli):
        """Find prototype nodes using the Minimum Spanning Tree (MST) approach.

        """

        logger.debug('Finding prototypes ...')

        start = time.time()


        """Questi 3 array li useremo durante la concorrenza
            P-> array dei pred
            C-> array dei costi
            U-> array degli used per vedere se gia' abbiamo usato il nodo
        """
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
        self.creaTagli(tagli, parti, self.subgraph.n_nodes)





        # While the heap is not empty

        while p != -1:

            """if flagTime:
                startPerc = time.time()
                flagTime = False"""

            # Remove a node from the heap


            # Gathers its cost from the heap
            self.subgraph.nodes[p].cost = C[p]
            U[p]=1
            #C[p]=

            # And also its predecessor

            pred= P[p]


            # If the predecessor is not NIL update prototypes
            if pred != c.NIL:
                self.updateProt(prototypes,p,pred)

            for i in range(tagli):
                work.put((parti[i][0], parti[i][1], p))

            work.join()


            # prendo il minimo
            p = self.calcMin(result)

            """percnew = (percent / self.subgraph.n_nodes) * 100


            if (percnew > percOld):
                endPerc = time.time()
                percOld += 1
                print("Prototypes %" + str(int(percnew)))
                print("Estimeted time: " + str((endPerc - startPerc) * (100 - percnew)) + " seconds")
                flagTime = True

            percent += 1"""
        work.join()

        for i in range(self._processi):
            processi[i].terminate()

        #Aggiorno il grafo
        for i in range(self.subgraph.n_nodes):
            self.subgraph.nodes[i].pred = P[i]

        end = time.time()
        fittime = end - start
        logger.debug('Prototypes: %s.', prototypes)
        logger.info('Prototypes found in: %s seconds.', fittime)


    def protParal(self,P, C, U, work, result):

        while True:
            r1,r2,p=work.get()

            s1=None

            for q in range(r1,r2):
                #Vedo se il nodo non è ancora stato usato
                if U[q] == 0:
                    # If `p` and `q` identifiers are different
                    if p != q:

                        weight=self.calcWeight(p,q)
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


    def initGraphFit(self,U,C,P,L):
        flag = True  # per prendere il primo prototipo

        # For each possible node
        for i in range(self.subgraph.n_nodes):
            U[i] = 0
            # Checks if node is a prototype
            if self.subgraph.nodes[i].status == c.PROTOTYPE:

                """Se e' un prototipo Costo=0, Pred=nil, Label=la stessa del nodo"""
                C[i] = 0
                P[i] = c.NIL
                L[i] = self.subgraph.nodes[i].label
                # prendo il primo prototipo
                if flag:
                    primo = i
                    flag = False

            # If node is not a prototype
            else:
                """se non è un prototipo Costo=MAX, Pred=lo stesso del nodo,  label=nil"""

                C[i] = c.FLOAT_MAX
                P[i] = self.subgraph.nodes[i].pred
                L[i] = c.NIL

        return primo

    def fitCompute(self,s,U,C,work,result,tagli,parti):
        """Percentuali per la stampa del tempo stimato e a che % stiamo"""
        percent = 0
        percOld = 1
        flagTime = True
        # quando avrò computato tutti i nodi s sarà = -1
        while s != -1:

            """Sempre per tempo e %"""
            """if flagTime:
                startPerc = time.time()
                flagTime = False"""

            # marchio il nodo come usato
            U[s] = 1

            # Lo aggiungo alla lista ordinata
            self.subgraph.idx_nodes.append(s)

            # Gathers its cost
            self.subgraph.nodes[s].cost = C[s]

            # Metto nella JoinableQueue tutte le partizioni e il nodo su cui operare
            for i in range(tagli):
                work.put((parti[i][0], parti[i][1], s))

            # Aspetto che la coda si svuota
            work.join()

            # prendo il minimo
            s = self.calcMin(result)

            # Tengo traccia a che punto sta, per file grossi è comodo
            """percnew = (percent / self.subgraph.n_nodes) * 100

            #Sempre per tenere traccia
            if (percnew > percOld):
                endPerc = time.time()
                percOld += 1
                print("Training %" + str(int(percnew)))
                print("Estimeted time: " + str((endPerc - startPerc) * (100 - percnew)) + " seconds")
                flagTime = True

            percent += 1"""

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
        self._find_prototypes(tagli)

        start = time.time()




        """Questi 3 array li useremo durante la concorrenza
                    P-> array dei pred
                    C-> array dei costi
                    U-> array degli used per vedere se gia' abbiamo usato il nodo
                    L-> array delle label
        """
        P=Array('i',self.subgraph.n_nodes,lock=False)
        C=Array('f',self.subgraph.n_nodes,lock=False)
        L=Array('i', self.subgraph.n_nodes,lock=False)
        U=Array('i',self.subgraph.n_nodes,lock=False)

        #Inizializzo gli array e prendo il primo prototipo
        primo=self.initGraphFit(U,C,P,L)


        processi = []

        work = JoinableQueue()
        result = Queue()

        #creo e faccio partire i processi
        self.creaProcFit(self.train,processi,P,C,L,U, work, result)


        """parti= [[0,n_nodi/tagli],...,[(tagli-1)*(n_nodi/tagli),n_nodi]]""" #partizionato in n parti uguali con n=tagli
        parti=[]
        self.creaTagli(tagli,parti, self.subgraph.n_nodes)



        # prendo il primo prototipo
        s = primo

        #Inizia il vero e proprio training
        self.fitCompute(s,U,C,work,result,tagli,parti)


        """Termino i processi"""
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



    def train(self, P,C,L,U, work, result):


        while True:
            #vedo se c'è un range sul quale lavorare
            r1,r2, s = work.get()

            # s1 dovrà essere il nodo non usato con il costo più piccolo, workInRange è proprio il lavoro del processo nel range r1,r2
            s1 = self.workInRange(s,r1,r2,C,L,P,U)


            # s1=None significa che ogni nodo del range di questo processo è già stato used
            if s1 == None:
                s1 = -1
            # restituisco il risultato al processo principale
            result.put((s1, C[s1]))
            #finisco su questo range
            work.task_done()

    def workInRange(self,s,r1,r2,C,L,P,U):
        s1=None
        # lavoro solo nel range preso dalla work.get() r1,r2
        for t in range(r1, r2):

            # se non stiamo confrontando lo stesso nodo con se stesso
            if s != t:

                # se il costo di s è più piccolo di t
                if C[t] > C[s]:

                    #calcolo la distanza tra s e t
                    weight = self.calcWeight(s, t)

                    # Il costo corrente sarà il massimo tra il costo dell'arco tra i due nodi (weight, l'arco in realtà non esiste) e il nodo s
                    current_cost = np.maximum(C[s], weight)

                    # If current cost is smaller than `q` node's cost
                    if current_cost < C[t]:
                        # aggiorno la label di t che sarà uguale a quella di s
                        L[t] = L[s]
                        # aggiorno il costo di t con quello corrente
                        C[t] = current_cost
                        P[t] = s

                # se s1 non è stato ancora assegnato oppure il costo di s1>costo t1
                if (s1 == None or C[s1] > C[t]):
                    # e se il nodo non è stato già used
                    if U[t] == 0:
                        s1 = t
        return s1

    def learn(self, xt, yt, xv, yv,tagli, n_iterations=10):
        """Learns the best classifier over a validation set.

        Args:
            xt (np.array): Array of training features.
            yt (np.array): Array of training labels.
            xv (np.array): Array of validation features.
            yv (np.array): Array of validation labels.
            n_iterations (int): Number of iterations.

        """

        X_val=copy.deepcopy(xv)
        X_train=copy.deepcopy(xt)
        Y_val=copy.deepcopy(yv)
        Y_train=copy.deepcopy(yt)

        logger.info('Learning the best classifier ...')

        # Defines the maximum accuracy
        max_acc = 0

        # Defines the previous accuracy
        previous_acc = 0

        # Defines the iterations counter
        t = 0

        # An always true loop
        while True:
            logger.info('Running iteration %d/%d ...', t+1, n_iterations)

            # Fits training data into the classifier
            self.fit(X_train, Y_train,tagli)

            # Predicts new data
            preds = self.pred(X_val,6)

            # Calculating accuracy
            acc = g.opf_accuracy(Y_val, preds)

            # Checks if current accuracy is better than the best one
            if acc > max_acc:
                # If yes, replace the maximum accuracy
                max_acc = acc

                # Makes a copy of the best OPF classifier
                best_opf = copy.deepcopy(self)

                xt[:]=X_train[:]
                xv[:]=X_val[:]
                yt[:]=Y_train[:]
                yv[:]=Y_val[:]
                # And saves the iteration number
                best_t = t

            # Gathers which samples were missclassified
            errors = np.argwhere(Y_val != preds)

            # Defining the initial number of non-prototypes as 0
            non_prototypes = 0

            # For every possible subgraph's node
            for n in self.subgraph.nodes:
                # If the node is not a prototype
                if n.status != c.PROTOTYPE:
                    # Increments the number of non-prototypes
                    non_prototypes += 1

            # For every possible error
            for err in errors:
                # Counter will receive the number of non-prototypes
                ctr = non_prototypes

                # While the counter is bigger than zero
                while ctr > 0:
                    # Generates a random index
                    j = int(r.generate_uniform_random_number(0, len(X_train)))

                    # If the node on that particular index is not a prototype
                    if self.subgraph.nodes[j].status != c.PROTOTYPE:
                        # Swap the input nodes
                        X_train[j, :], X_val[err, :] = X_val[err, :], X_train[j, :]

                        # Swap the target nodes
                        Y_train[j], Y_val[err] = Y_val[err], Y_train[j]

                        # Decrements the number of non-prototypes
                        non_prototypes -= 1

                        # Resets the counter
                        ctr = 0

                    # If the node on that particular index is a prototype
                    else:
                        # Decrements the counter
                        ctr -= 1

            # Calculating difference between current accuracy and previous one
            delta = np.fabs(acc - previous_acc)

            # Replacing the previous accuracy as current accuracy
            previous_acc = acc

            # Incrementing the counter
            t += 1

            logger.info('Accuracy: %s | Delta: %s | Maximum Accuracy: %s', acc, delta, max_acc)

            # If the difference is smaller than 10e-4 or iterations are finished
            if delta < 0.0001 or t == n_iterations:
                # Replaces current class with the best OPF
                self.subgraph = best_opf.subgraph
                self.pred(xt,6)

                # Breaks the loop
                break

        #self.predict(xv)

        """# Calculating accuracy
        acc = g.opf_accuracy(yv, preds)
        print("ecco2 ", acc)"""


        return max_acc


    def prune(self, X_train, Y_train, X_val, Y_val, tagli,M_loss, n_iterations=10):
        """Prunes a classifier over a validation set.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            n_iterations (int): Maximum number of iterations of learning.

        """


        logger.info('Pruning classifier ...')

        # Faccio il primo learning e mi calcolo l'accuratezza massima
        acc=self.learn(X_train, Y_train, X_val, Y_val,tagli, n_iterations=n_iterations)

        #Prendo i nodi iniziali del grafo
        initial_nodes = self.subgraph.n_nodes

        # Faccio partire il pruring
        self.pruringRun(acc,M_loss,tagli,n_iterations,X_train, Y_train, X_val, Y_val)


        #Prendo i nodi a fine pruring
        final_nodes = self.subgraph.n_nodes
        logger.info('Initial number of nodes: %s , final number of nodes: %s,',initial_nodes,final_nodes)
        # Calculating pruning ratio
        prune_ratio = 1 - final_nodes / initial_nodes

        logger.info('Prune ratio: %s.', prune_ratio)


    def pruringRun(self,acc,M_loss,tagli,n_iterations,X_train, Y_train, X_val, Y_val):
        tmp = acc
        flag = True

        #mentre l'accuratezza attuale è >= (dell'accuratezza massima iniziale - M_loss) e mentre ci sta almeno un nodo non Rilevante (flag)
        while abs(acc-tmp) <= M_loss and flag:
            #Rimuovo i nodi irrillevanti, aggiorno X_train, Y_train, X_val,Y_val e vedo se ci sta almeno un nodo non rilevante con flag
            flag,X_train, Y_train, X_val,Y_val =self.pruringUpdateList(X_train,Y_train,X_val,Y_val)

            #Faccio il learning e prendo l'accuratezza
            tmp=self.learn(X_train, Y_train, X_val, Y_val, tagli, n_iterations=n_iterations)
            logger.info('Current accuracy: %s.', tmp)

    def pruringUpdateList(self,X_train,Y_train,X_val,Y_val):

        #Liste temporanee per X_train, Y_trrain, X_val e Y_val
        X_temp, Y_temp = [], []
        X_t, Y_t = [], []

        #Se flag rimanse false significa che tutti i nodi sono rilevanti e quindi possiamo terminare il pruning
        flag = False
        for j, n in enumerate(self.subgraph.nodes):
            #Aggiungo alle liste temporanee per X_train, Y_train i nodi rilevanti
            if n.relevant != c.IRRELEVANT:
                X_temp.append(X_train[j, :])
                Y_temp.append(Y_train[j])
            #Aggiungo alle liste temporanee per X_val e Y_val i nodi non rilevanti
            else:
                #Non tutti i nodi sono rilevanti
                flag = True
                X_t.append(X_train[j, :])
                Y_t.append(Y_train[j])
        #Infine aggiungo alle liste temporanee per X_val e Y_val i nodi di testing che già c'erano
        for j in range(len(Y_val)):
            X_t.append(X_val[j])
            Y_t.append(Y_val[j])

        #restituisco tutti i numpy array
        return flag,np.asarray(X_temp),np.asarray(Y_temp),np.asarray(X_t), np.asarray(Y_t)


    def predConc(self, work,X_val, result,conquerors):

        while True:
            ran =work.get()
            pred= self.predict(X_val[ran[0]:ran[1]], coda=conquerors)

            j=0
            for i in range(ran[0],ran[1]):
                result[i]=pred[j]
                j+=1

            work.task_done()


    def pred(self, X_val,tagli, I_val=None):
        #tagli
        t=[]
        #processi
        p=[]

        self.creaTagli(tagli,t,len(X_val))
        work=JoinableQueue()
        conquerors=Queue()

        #ci vanno i risultati in ordine
        result=Array('i',len(X_val),lock=False)

        self.creaProcFit(self.predConc,p,work,X_val,result,conquerors)

        for i in range(len(t)):
            work.put(t[i])

        work.join()

        for i in range(self._processi):
            p[i].terminate()


        while not conquerors.empty():
            self.subgraph.mark_nodes(conquerors.get())

        return result



    def predict(self, X_val, I_val=None, coda=None):
        """Predicts new data using the pre-trained classifier.

        Args:
            X_val (np.array): Array of validation or test features.
            I_val (np.array): Array of validation or test indexes.

        Returns:
            A list of predictions for each record of the data.

        """

        # Checks if there is a subgraph
        if not self.subgraph:
            # If not, raises an BuildError
            raise e.BuildError('Subgraph has not been properly created')

        # Checks if subgraph has been properly trained
        if not self.subgraph.trained:
            # If not, raises an BuildError
            raise e.BuildError('Classifier has not been properly fitted')

        logger.info('Predicting data ...')

        # Initializing the timer
        start = time.time()

        # Creating a prediction subgraph
        pred_subgraph = Subgraph(X_val, I=I_val)

        # For every possible node
        for i in range(pred_subgraph.n_nodes):
            # Initializing the conqueror node
            conqueror = -1

            # Initializes the `j` counter
            j = 0

            # Gathers the first node from the ordered list
            k = self.subgraph.idx_nodes[j]

            # Checks if we are using a pre-computed distance
            if self.pre_computed_distance:
                # Gathers the distance from the distance's matrix
                weight = self.pre_distances[self.subgraph.nodes[k].idx][pred_subgraph.nodes[i].idx]

            # If the distance is supposed to be calculated
            else:
                # Calls the corresponding distance function
                weight = self.distance_fn(
                    self.subgraph.nodes[k].features, pred_subgraph.nodes[i].features)

            # The minimum cost will be the maximum between the `k` node cost and its weight (arc)
            min_cost = np.maximum(self.subgraph.nodes[k].cost, weight)

            # The current label will be `k` node's predicted label
            current_label = self.subgraph.nodes[k].predicted_label

            # While `j` is a possible node and the minimum cost is bigger than the current node's cost
            while j < (self.subgraph.n_nodes - 1) and min_cost > self.subgraph.nodes[self.subgraph.idx_nodes[j+1]].cost:
                # Gathers the next node from the ordered list
                l = self.subgraph.idx_nodes[j+1]

                # Checks if we are using a pre-computed distance
                if self.pre_computed_distance:
                    # Gathers the distance from the distance's matrix
                    weight = self.pre_distances[self.subgraph.nodes[l].idx][pred_subgraph.nodes[i].idx]

                # If the distance is supposed to be calculated
                else:
                    # Calls the corresponding distance function
                    weight = self.distance_fn(
                        self.subgraph.nodes[l].features, pred_subgraph.nodes[i].features)

                # The temporary minimum cost will be the maximum between the `l` node cost and its weight (arc)
                temp_min_cost = np.maximum(self.subgraph.nodes[l].cost, weight)

                # If temporary minimum cost is smaller than the minimum cost
                if temp_min_cost < min_cost:
                    # Replaces the minimum cost
                    min_cost = temp_min_cost

                    # Gathers the identifier of `l` node
                    conqueror = l

                    # Updates the current label as `l` node's predicted label
                    current_label = self.subgraph.nodes[l].predicted_label

                # Increments the `j` counter
                j += 1

                # Makes `k` and `l` equals
                k = l

            # Node's `i` predicted label is the same as current label
            pred_subgraph.nodes[i].predicted_label = current_label

            # Checks if any node has been conquered
            if conqueror > -1:
                # Marks the conqueror node and its path
                if coda is not None:
                    coda.put(conqueror)
                else:
                    self.subgraph.mark_nodes(conqueror)

        # Creating the list of predictions
        preds = [pred.predicted_label for pred in pred_subgraph.nodes]

        # Ending timer
        end = time.time()

        # Calculating prediction task time
        predict_time = end - start

        logger.info('Data has been predicted.')
        logger.info('Prediction time: %s seconds.', predict_time)


        return preds
