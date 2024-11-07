import csv
import numpy as np

pathTable = "D:/PraktikumUniKlinik/data/dataSlide/Statistik/Test.csv"

def csvToList(path):
    list = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            list.append(row[0].split(','))
    return list

def ListCaluclate(list):
    listtypes = list[0][3:]
    counts = np.zeros(len(listtypes))
    countsTrues = np.zeros(len(listtypes))
    countsFalses = np.zeros(len(listtypes))
    percentsTypes = np.zeros(len(listtypes))

    for i in list[1:]:
        for il,l in enumerate(listtypes):
            if l == i[1]:
                counts[il] += 1
                if i[1] == i[2]:
                    countsTrues[il] += 1
                else:
                    countsFalses[il] += 1

    for il, l in enumerate(listtypes):
        percentsTypes[il] = countsTrues[il] / counts[il]

    return listtypes, np.array([counts, countsTrues, countsFalses]), percentsTypes



if __name__ == "__main__":
    list = csvToList(pathTable)
    listTypName, listCounts, percentsTypes = ListCaluclate(list)
    print(listTypName)
    print(listCounts)
    print(percentsTypes)