"""Supervised Optimum-Path Forest.
"""

import copy
import time

import numpy as np

import opfython.math.general as g
import opfython.math.random as r
import opfython.utils.constants as c
import opfython.utils.exception as e
import opfython.utils.logging as log
from opfython.core import OPF, Heap, Subgraph

from multiprocessing import JoinableQueue, Process,Queue


logger = log.get_logger(__name__)


class SSupervisedPOPF(OPF):
    """A SupervisedOPF which implements the supervised version of OPF classifier.

    References:
        J. P. Papa, A. X. Falcão and C. T. N. Suzuki. Supervised Pattern Classification based on Optimum-Path Forest.
        International Journal of Imaging Systems and Technology (2009).

    """

    def __init__(self,processi=4, distance='log_squared_euclidean', pre_computed_distance=None):
        """Initialization method.

        Args:
            distance (str): An indicator of the distance metric to be used.
            pre_computed_distance (str): A pre-computed distance file for feeding into OPF.

        """

        logger.info('Overriding class: OPF -> SupervisedOPF.')

        # Override its parent class with the receiving arguments
        super(SSupervisedPOPF, self).__init__(distance, pre_computed_distance)

        self._processi=processi

        logger.info('Class overrided.')

    def _find_prototypes(self):
        """Find prototype nodes using the Minimum Spanning Tree (MST) approach.

        """

        logger.debug('Finding prototypes ...')

        start=time.time()

        # Creating a Heap of size equals to number of nodes
        h = Heap(self.subgraph.n_nodes)

        # Marking first node without any predecessor
        self.subgraph.nodes[0].pred = c.NIL

        # Adding first node to the heap
        h.insert(0)

        # Creating a list of prototype nodes
        prototypes = []

        # While the heap is not empty
        while not h.is_empty():
            # Remove a node from the heap
            p = h.remove()

            # Gathers its cost from the heap
            self.subgraph.nodes[p].cost = h.cost[p]

            # And also its predecessor
            pred = self.subgraph.nodes[p].pred

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

            # For every possible node
            for q in range(self.subgraph.n_nodes):
                # Checks if the color of current node in the heap is not black
                if h.color[q] != c.BLACK:
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
                        if weight < h.cost[q]:
                            # Marks `q` predecessor node as `p`
                            self.subgraph.nodes[q].pred = p

                            # Updates the arc on the heap
                            h.update(q, weight)
        end= time.time()
        fittime=end-start
        logger.debug('Prototypes: %s.', prototypes)
        logger.info('Prototypes found in: %s seconds.', fittime)


    def fit(self, X_train, Y_train, I_train=None):
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
        self._find_prototypes()

        start = time.time()

        """partizione è fatta ccosì:[used, cost, pred, label] 
            Used=se il nodo è stato già usato
            Cost=costo del nodo
            Pred=predecessore del nodo
            Label=a quale label si riferisce
        """
        partizione=[]


        flag=True #per prendere il primo prototipo


        # For each possible node
        for i in range(self.subgraph.n_nodes):
            # Checks if node is a prototype
            if self.subgraph.nodes[i].status == c.PROTOTYPE:
                # If yes, it does not have predecessor nodes
                self.subgraph.nodes[i].pred = c.NIL
                # Its predicted label is the same as its true label
                self.subgraph.nodes[i].predicted_label = self.subgraph.nodes[i].label


                #per ogni prototipo usato=0 costo=0 pres=nil label=la label a quale il nodo già si riferisce
                partizione.append([0,0,c.NIL,self.subgraph.nodes[i].label])
                if flag:
                    primo=i
                    flag=False

            # If node is not a prototype
            else:
                #se non è un prototipo usato=0 cosot=MAX pred=quello che già stava nel label=nil
                partizione.append([0, c.FLOAT_MAX, self.subgraph.nodes[i].pred,''])


        """###########################################################################################################################
        #Creo i processi
            
            processi= gli m processi che usiamo saranno memorizzati qui!
            
            work= Coda dalla quale i processi prenderanno informazioni dal processo principale ----> (partizione[s],s)
                partizione[s]=le informazioni [usato,costo,pred,label] di un nodo s
                s=nodo (int)
            
            result= sono i risultati parziali per ogni iterazione del while di ogni processo ----> (s1,partizione[s1][1])
                s1=il nodo(non used) con il costo più piccolo
                partizione[s1][1]= il suo rispettivo costo per aggiornalo nel processo principale
            
            result è una JoinableQueue per garantire che ogni processo non acceda a work due volte consecutive prima dei prossimi put del processo principale
            
            finalResult=I risultati finali di ogni processo, servono per aggiornare il costo dopo il quale il processo termina   ----> (finalResult)
                finalResult= dizionario {t: (s,partizione[s][3])} ogni processo ogni volta che aggiorna partizione[t], fa l'update del dizionario per aggiornare  il nuovo pred di t e 
                                                                                                                                                    la nuova label di t
                
                        
        
        """
        processi=[]

        work=Queue()
        result=JoinableQueue()
        finalResult=Queue()
        for i in range(self._processi):
            processi.append(Process(target=self.train, args=(work,result,i,finalResult,partizione)))
            processi[i].daemon=True
            processi[i].start()


        """"###########################################################################################################################
            ###########################################################################################################################
            ###########################################################################################################################
            ###########################################################################################################################   
        """


        #prendo il primo prototipo
        s=primo




        # quando avrò computato tutti i nodi s sarà = -1
        while s!=-1:



            #marchio il nodo come usato
            partizione[s][0]=1

            # Appends its index to the ordered list
            self.subgraph.idx_nodes.append(s)

            # Gathers its cost
            self.subgraph.nodes[s].cost = partizione[s][1]


            #Mando ad ogni processo s e i suoi dati contenuti in partizione[s]
            for _ in range(self._processi):
                work.put((partizione[s],s))


            #Aspetto i risultati parziali di ogni processo
            risultati=[]
            for _ in range(self._processi):
                risultati.append(result.get())

                #All'ultima iterazione del for i processi veranno sbloccati e possono di nuovo attendere in work.get() per i nuovi compiti
                result.task_done()


            #ora devo prendere il minimo in termini di costo dei risultati parziali
            #min --> (index, costo)
            min=risultati[0]

            j=0
            # questo for serve per togliere un errore che non capita quasi mai!
            for i in range(1,self._processi):
                if min[0]==-1 and risultati[i][0]!=-1:
                    min=risultati[i]
                    j=i
                    break

            #cerco il minimo se esiste
            for i in range(j, self._processi):
                if min[1]>risultati[i][1] and risultati[i][0]!=-1:
                    min=risultati[i]
            #prendo il minimo
            s=min[0]

            #s=-1 significa che ho computato tutti i nodi altrimenti aggiorno il costo di s
            if s!=-1:
             partizione[s][1]=min[1]




        #Sono uscito dal while, devo terminare i Processi
        for _ in range(self._processi):
            work.put((None, None))

        #dizionario che serve per aggiornare le informazioni del grafo la sua struttura sta sopra
        res={}
        #prendo tutti i risulati finali dei processi (il loro ultimo lavoro)
        for i in range(self._processi):
            res.update(finalResult.get())

        #aggiorno pred e label dei nodi
        """Purtroppo questa operazione è neccessaria perché non si può mandare ai processi oggetti per riferimento O((n^2/m)+n).
           Si potrebbe usare il BaseManager e non avere il +n finale però avendo già precedentemente implementato lo stesso codice con il BaseManager l'efficienza senza
           è nettamente superiore che con, in termini di velocità."""
        for key in res:

            self.subgraph.nodes[key].pred = res[key][0]
            self.subgraph.nodes[key].predicted_label = res[key][1]


        # The subgraph has been properly trained
        self.subgraph.trained = True

        # Ending timer
        end = time.time()

        # Calculating training task time
        train_time = end - start

        logger.info('Classifier has been fitted.')
        logger.info('Training time: %s seconds.', train_time)


    def train(self,work,result,i,finalResultRet,partizione):

        #Calcolo il range di azione (r1,r2) del processo in base a quanti processi ci sono
        if i == 0:
            r1 = 0
        else:
            r1 = i *int( (self.subgraph.n_nodes/ self._processi))
        if i == self._processi - 1:
            r2 = self.subgraph.n_nodes
        else:
            r2 = int(r1 + (self.subgraph.n_nodes/ self._processi))


        #qui andranno i risultati finali
        finalResult={}


        while True:

            #Attendo parts=partizione[s] ed s
            parts,s=work.get()


            #Se parts=None significa che il lavoro è finito e posso restituire finalResult
            if parts==None:

               finalResultRet.put(finalResult)
               return

            #Marchio s come used
            partizione[s][0] = 1

            #s1 dovrà essere il nodo non usato con il costo più piccolo
            s1=None

            #lavoro solo nel mio range
            for t in range(r1,r2):

                #se non stiamo confrontando lo stesso nodo con se stesso
                if s!=t:

                    #se il costo di s è più piccolo di t
                    if partizione[t][1]> parts[1]:

                        if self.pre_computed_distance:
                            # Gathers the distance from the distance's matrix
                            weight = self.pre_distances[self.subgraph.nodes[s].idx][self.subgraph.nodes[t].idx]

                        # If the distance is supposed to be calculated
                        else:
                            # Calls the corresponding distance function
                            weight = self.distance_fn(
                                self.subgraph.nodes[t].features, self.subgraph.nodes[s].features)




                        #Il costo corrente sarà il massimo tra il costo dell'arco tra i due nodi (weight, l'arco in realtà non esiste) e il nodo s
                        current_cost = np.maximum(partizione[s][1], weight)

                        # If current cost is smaller than `q` node's cost
                        if current_cost < partizione[t][1]:

                            #aggiorno la label di t che sarà uguale a quella di s
                            partizione[t][3] = partizione[s][3]
                            #aggiorno il costo di t con quello corrente
                            partizione[t][1] = current_cost

                            #aggiorno anche finalResult che ci servirà alla fine
                            finalResult.update({t: (s, partizione[s][3])})

                    #se s1 non è stato ancora assegnato oppure il costo di s1>costo t1
                    if (s1==None or partizione[s1][1]>partizione[t][1]):
                        #e se il nodo non è stato già used
                        if partizione[t][0]==0:
                            s1=t


            #s1=None significa che ogni nodo del range di questo processo è già stato used
            if s1==None:
                s1=-1
            #restituisco il risultato al processo principale
            result.put((s1,partizione[s1][1]))
            #aspetto che il processo principale abbia finito(significa che anche gli altri processi hanno finito) così da non poter rubare il work di un processo che non ha ancora fatto la work.get()
            result.join()




    def predict(self, X_val, I_val=None):
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

    def learn(self, X_train, Y_train, X_val, Y_val, n_iterations=10):
        """Learns the best classifier over a validation set.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            n_iterations (int): Number of iterations.

        """

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
            self.fit(X_train, Y_train)

            # Predicts new data
            preds = self.predict(X_val)

            # Calculating accuracy
            acc = g.opf_accuracy(Y_val, preds)

            # Checks if current accuracy is better than the best one
            if acc > max_acc:
                # If yes, replace the maximum accuracy
                max_acc = acc

                # Makes a copy of the best OPF classifier
                best_opf = copy.deepcopy(self)

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
                self = best_opf

                logger.info('Best classifier has been learned over iteration %d.', best_t+1)

                # Breaks the loop
                break

    def prune(self, X_train, Y_train, X_val, Y_val, n_iterations=10):
        """Prunes a classifier over a validation set.

        Args:
            X_train (np.array): Array of training features.
            Y_train (np.array): Array of training labels.
            X_val (np.array): Array of validation features.
            Y_val (np.array): Array of validation labels.
            n_iterations (int): Maximum number of iterations.

        """

        logger.info('Pruning classifier ...')

        # Fits training data into the classifier
        self.fit(X_train, Y_train)

        # Predicts new data
        self.predict(X_val)

        # Gathering initial number of nodes
        initial_nodes = self.subgraph.n_nodes

        # For every possible iteration
        for t in range(n_iterations):
            logger.info('Running iteration %d/%d ...', t+1, n_iterations)

            # Creating temporary lists
            X_temp, Y_temp = [], []

            # Removing irrelevant nodes
            for j, n in enumerate(self.subgraph.nodes):
                if n.relevant != c.IRRELEVANT:
                    X_temp.append(X_train[j, :])
                    Y_temp.append(Y_train[j])

            # Copying lists back to original data
            X_train = np.asarray(X_temp)
            Y_train = np.asarray(Y_temp)

            # Fits training data into the classifier
            self.fit(X_train, Y_train)

            # Predicts new data
            preds = self.predict(X_val)

            # Calculating accuracy
            acc = g.opf_accuracy(Y_val, preds)

            logger.info('Current accuracy: %s.', acc)

        # Gathering final number of nodes
        final_nodes = self.subgraph.n_nodes

        # Calculating pruning ratio
        prune_ratio = 1 - final_nodes / initial_nodes

        logger.info('Prune ratio: %s.', prune_ratio)
