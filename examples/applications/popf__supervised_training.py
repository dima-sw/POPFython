import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.POPF__supervised import SSupervisedPOPF
import multiprocessing as mp
import time



if __name__ == '__main__':


    tagli = 10
    # Loading a .txt file to a numpy array
    txt = l.load_txt(r'C:\Users\TheDimitri\Documents\GitHub\POPFython\data\nug.txt')

    # Parsing a pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Splitting data into training and testing sets
    X_train, X_test, Y_train, Y_test = s.split(
        X, Y, percentage=0.5, random_state=1)



    # Creates a SupervisedPOPF instance
    opf = SSupervisedPOPF(processi=mp.cpu_count(),tagli=tagli
                             ,distance='log_squared_euclidean',
                            pre_computed_distance=None)

    t1 = time.time()

    opf.fit(X_train, Y_train)
    preds=opf.pred(X_test)

    acc = g.opf_accuracy(Y_test, preds)

    t2 = time.time()
    tot = t2 - t1

    print("Tempo: ", tot)
    print(f'Accuracy: {acc}')