
import numpy as np
import copy
import opfython.math.random as r
import opfython.utils.constants as c
import opfython.utils.logging as log
import opfython.math.general as g

logger = log.get_logger(__name__)


def learn(self, xt, yt, xv, yv, tagli, n_iterations=10):
    """Learns the best classifier over a validation set.
    Args:
        xt (np.array): Array of training features.
        yt (np.array): Array of training labels.
        xv (np.array): Array of validation features.
        yv (np.array): Array of validation labels.
        n_iterations (int): Number of iterations.
    """
    # Devo salvare i numpyArray del grafo con l'accuratezza maggiore
    X_val = copy.deepcopy(xv)
    X_train = copy.deepcopy(xt)
    Y_val = copy.deepcopy(yv)
    Y_train = copy.deepcopy(yt)

    logger.info('Learning the best classifier ...')

    # Defines the maximum accuracy
    max_acc = 0

    # Defines the previous accuracy
    previous_acc = 0

    # Defines the iterations counter
    t = 0

    # An always true loop
    while True:
        logger.info('Running iteration %d/%d ...', t + 1, n_iterations)

        # Fits training data into the classifier
        self.fit(X_train, Y_train, tagli)

        # Predicts new data
        # preds = self.pred(X_val,tagli)
        preds = self.pred(X_val, tagli)
        # Calculating accuracy
        acc = g.opf_accuracy(Y_val, preds)

        # Checks if current accuracy is better than the best one
        if acc > max_acc:
            # If yes, replace the maximum accuracy
            max_acc = acc

            # Makes a copy of the best OPF classifier
            best_opf = copy.deepcopy(self)

            # Salvo i numpyArray del classificatore con l'accuratezza maggiore
            xt[:] = X_train[:]
            xv[:] = X_val[:]
            yt[:] = Y_train[:]
            yv[:] = Y_val[:]
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
            self.pred(X_val, tagli)
            # Breaks the loop
            break
    print(max_acc)
    return max_acc