import random
"""

Create your own random DATASET!! Number of features:3, number of sample: your choise

@Author: Dmytro Lysyhanych

"""
def creaGig(n):
    giganti=open("../../data/nug.txt", "w")
    #altezza, età, peso,
    carU=[(150,190),(19,50),(55,90)]
    carG = [(145, 175), (51, 120), (45, 70)]
    carN = [(44, 128), (0, 11), (6, 24)]
    carR=[(126, 180), (11, 18), (26, 87)]
    tutto=[carN,carR,carU,carG]



    for j in range(n):
        #for j in range(2000):

            i= random.randint(0,3)

            giganti.write(str(j)+" "+str(i+1)+" "+str(random.uniform(tutto[i][0][0],tutto[i][0][1]))+" "+str(random.uniform(tutto[i][1][0],tutto[i][1][1]))+" "+str(random.uniform(tutto[i][2][0],tutto[i][2][1]))+'\n')

    giganti.close()

creaGig(100000)

