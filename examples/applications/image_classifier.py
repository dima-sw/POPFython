import opfython.math.general as g
import opfython.stream.loader as l
import opfython.stream.parser as p
import opfython.stream.splitter as s
from opfython.stream.splitter import split
from opfython.models.POPF__supervised import SSupervisedPOPF
import multiprocessing as mp


import logging,os,time
import dill
from IMGToData.IMGToNumpy import convertIMG,takeImgNP



def caricaFile(pathIMG,dictIMG):

    quanti = input("Quanti animali vogliamo classificare? ")
    quanteCartelle = input("Quante cartelle")

    convertIMG(quanti, quanteCartelle, 1, 28, pathIMG, dictIMG)
    return quanteCartelle


def loadOpf():
    try:

        with open('opf.pkl', 'rb') as f:
            opf = dill.load(f)

        return opf

    except:
        print("There is not opf saved")

def saveOPF(opf):
    if os.path.exists("opf.pkl"):
        os.remove("opf.pkl")
    with open('opf.pkl', 'wb') as output:
        dill.dump(opf, output)


def loadDataFromFile(filePath):
    txt = l.load_txt(filePath)
    X, Y = p.parse_loader(txt)
    X_train, X_test, Y_train, Y_test = split(
        X, Y, percentage=0.9, random_state=1)

    return X_train, X_test, Y_train, Y_test


def createOPF(cpu):
    opf = SSupervisedPOPF(processi=cpu
                          , distance='log_squared_euclidean',
                          pre_computed_distance=None)
    return opf

def trainOPF(opf,X_train,Y_train):
    opf.fit(X_train, Y_train,8)


def predictOPF(opf,X_test,Y_test):
    preds = opf.predict(X_test)

    # Calculating accuracy
    acc = g.opf_accuracy(Y_test, preds)

    print(f'Accuracy: {acc}')


def predictSample(opf,n_Samples):
    while True:

        #altezza età peso

        path=input("Inserisci path")

        try:

            np=takeImgNP(path,28)

            preds= opf.predict(np)

            print(preds)

            ris=list()

            for k in range(int(n_Samples)):
                ris.append(0)

            sum=0
            for s in range(len(preds)):

                if s==0 or s==6:
                    ris[preds[s]-1]+=3
                    sum+=3
                if s==1 or s==7:
                    ris[preds[s]-1] += 2
                    sum += 2
                else:
                    ris[preds[s]-1]+=1
                    sum+=1

            print("Metodo1: ")
            for s in range(len(ris)):
                print("Secondo me è " + str(dictIMG[s+1])+" per il: "+str((ris[s]/sum)*100)+" %")
            print("Metodo2:")
            rise=[]
            for k in range(int(n_Samples)):
                rise.append(0)

            sum=0
            for s in range(len(preds)):
                rise[preds[s]-1]+=1
                sum+=1

            for s in range(len(rise)):

                print("Secondo me è " + str(dictIMG[s+1])+" per il: "+str((rise[s]/sum)*100)+" %")

            """print("Metodo3: (Minore è meglio)")
            for k in range(int(n_Samples)):
                ris[k]=0

            for s in range(len(preds)):
                c = rise[preds[s] - 1]
                if c != 0:
                    ris[preds[s]-1]+=distances[s]/(c)


                    sum+=(distances[s]/(c))

            
            for s in range(len(ris)):
                if rise[s]==0:
                    print("E' altamente impossibile che sia: " + str(dictIMG[s + 1]))
                else:
                    print("Secondo me la differenza con: " + str(dictIMG[s+1])+" è per lo più: "+str(((ris[s])/sum)*100)+" %")

            print("Metodo4: (Minore è meglio)")
            for k in range(int(n_Samples)):
                ris[k]=0

            for s in range(len(preds)):
                c = rise[preds[s] - 1]
                if s == 0 or s == 6:
                    if c != 0:
                        ris[preds[s] - 1] += distances[s] / (c * c)

                        sum += (distances[s] / (c * c))
                elif s == 1 or s == 7:
                    if c != 0:
                        ris[preds[s] - 1] += distances[s] / (c * c)


                        sum += (distances[s] / (c * c))
                else:
                    if c != 0:
                        ris[preds[s] - 1] += distances[s]/c
                        c = rise[preds[s] - 1]

                        sum += (distances[s] / (c))


            
            for s in range(len(ris)):
                if rise[s]==0:
                    print("E' altamente impossibile che sia: " + str(dictIMG[s + 1]))

                else:
                    print("Secondo me la differenza con: " + str(dictIMG[s+1])+" è per lo più: "+str(((ris[s])/sum)*100)+" %")"""



        except Exception as e:
            print(e)




if __name__ == '__main__':
    # Loading a .txt file to a numpy array

    pathIMG="/home/dima/Desktop/raw-img"
    dictIMG={1: "cane", 2: "gatto", 3: "farfalla",4: "gallina",5: "elefante",6: "mucca",7: "pecora",8: "ragno",9: "scoiattolo"}

    filePath='/home/dima/Desktop/opfython-master/data/animal.txt'
    processi=mp.cpu_count()

    while True:
        try:
            i=input("Menu 1)Carica file\n2)Load data from file\n3)Create opf\n4)Load opf\n5)Save opf\n6)Fit opf\n7)Accuracy OPF\n8)Predict Samples :")
            inp=int(i)

            if inp==1:
                n_Samples=caricaFile(pathIMG,dictIMG)
            elif inp==2:
                X_train, X_test, Y_train, Y_test= loadDataFromFile(filePath)
            elif inp ==3:
                opf=createOPF(processi)
                print(opf)
            elif inp ==4:
                opf=loadOpf()
            elif inp ==5:
                saveOPF(opf)

            elif inp ==6:
                trainOPF(opf,X_train,Y_train)
            elif inp ==7:
                predictOPF(opf,X_test,Y_test)
            elif inp ==8:
                n_Samples=input("Qunati samples")
                predictSample(opf,n_Samples)
            else:
                print("Non ci sono altri opzioni :(")
        except Exception as e:
            logging.error(e)





"/home/dima/Desktop/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg"

