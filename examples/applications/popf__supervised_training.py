import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models.POPF__supervised import SSupervisedPOPF
import multiprocessing as mp
if __name__ == '__main__':
    # Loading a .txt file to a numpy array
    txt = l.load_txt('/home/dima/Desktop/opfython-master/data/nug.txt')

    # Parsing a pre-loaded numpy array
    X, Y = p.parse_loader(txt)

    # Splitting data into training and testing sets
    X_train, X_test, Y_train, Y_test = s.split(
        X, Y, percentage=0.5, random_state=1)


    print(mp.cpu_count())
    # Creates a SupervisedOPF instance
    opf = SSupervisedPOPF(processi=mp.cpu_count()
                         ,distance='log_squared_euclidean',
                        pre_computed_distance=None)




    # Fits training data into the classifier
    opf.fit(X_train, Y_train,8)





    # Predicts new data
    preds = opf.predict(X_test)

    # Calculating accuracy
    acc = g.opf_accuracy(Y_test, preds)


    print(f'Accuracy: {acc}')
