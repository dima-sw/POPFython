import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.models import SupervisedOPF
import time


# Loading a .txt file to a numpy array
txt = l.load_txt(r'C:\Users\TheDimitri\Documents\GitHub\POPFython\data\nug.txt')

# Parsing a pre-loaded numpy array
X, Y = p.parse_loader(txt)

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = s.split(
    X, Y, percentage=0.5, random_state=1)

# Creates a SupervisedOPF instance
opf = SupervisedOPF(distance='log_squared_euclidean',
                    pre_computed_distance=None)

t1=time.time()
# Fits training data into the classifier
opf.fit(X_train, Y_train)

# Predicts new data
preds = opf.predict(X_test)

# Calculating accuracy
acc = g.opf_accuracy(Y_test, preds)

t2=time.time()
print("Tempo: ", str(t2-t1))

print(f'Accuracy: {acc}')