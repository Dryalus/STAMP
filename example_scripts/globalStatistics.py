import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import csv
import os

path = "test"

def addDicEntry(dic, key, entry):
    if key in dic:
        dic[key] += [entry]
    else:
        dic[key] = [entry]
    return dic
def sortToDics(valueArray):
    def dicEntryToNumpyarray(dic):
        vkeys = dic.keys()
        for key in vkeys:
            dic[key] = np.array(dic[key])
        return dic

    aucScoreMean = {}
    aucScoreLow = {}
    aucScoreHigh = {}
    avgPrecScoreMean = {}
    avgPrecScoreLow = {}
    avgPrecScoreHigh = {}

    for i,value in enumerate(valueArray):
        vkeys = list(value.keys())
        for v in vkeys:
            x = np.transpose(np.array(value[v])) # Change Values to own list
            aucScoreMean = addDicEntry(aucScoreMean, v, x[0])
            aucScoreLow = addDicEntry(aucScoreLow, v, x[1])
            aucScoreHigh = addDicEntry(aucScoreHigh, v, x[2])
            avgPrecScoreMean = addDicEntry(avgPrecScoreMean, v, x[3])
            avgPrecScoreLow = addDicEntry(avgPrecScoreLow, v, x[4])
            avgPrecScoreHigh = addDicEntry(avgPrecScoreHigh, v, x[5])

    aucScoreMean = dicEntryToNumpyarray(aucScoreMean)
    aucScoreLow = dicEntryToNumpyarray(aucScoreLow)
    aucScoreHigh = dicEntryToNumpyarray(aucScoreHigh)
    avgPrecScoreMean = dicEntryToNumpyarray(avgPrecScoreMean)
    avgPrecScoreLow = dicEntryToNumpyarray(avgPrecScoreLow)
    avgPrecScoreHigh = dicEntryToNumpyarray(avgPrecScoreHigh)

    return aucScoreMean, aucScoreLow, aucScoreHigh, avgPrecScoreMean, avgPrecScoreLow, avgPrecScoreHigh

def getFileStrukture(path):
    dirNmax = os.listdir(path)
    dirArray = []
    listNmax = []
    listdecay = []
    valueArray = []
    for i, item in enumerate(dirNmax):
        listNmax.append(float("0." + item.split('ms')[1]))
        dirdecay = os.listdir(path+'/'+item)
        dirArray.append(dirdecay)
        valueList = {}
        for j in dirdecay:
            csvPath = path+'/'+item+'/'+j
            vecNames, arrayValues = csvToArray(csvPath)
            for vi, v in enumerate(vecNames):
                if v in valueList:
                    valueList[v] += [arrayValues[vi]]
                else:
                    valueList[v] = [arrayValues[vi]]
        valueArray.append(valueList)
    listNmax = np.array(listNmax)

    for item in dirArray[0]:
        listdecay.append(float("0." + item.split('model_statistics')[1]))
    listdecay = np.array(listdecay)

    return listNmax, listdecay, valueArray

def csvToArray(path):
    path = path + "/ConSub-categorical-stats-aggregated.csv"
    listarry = []
    vecNames = []
    arrayValues = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            listarry.append(row)
        for i, item in enumerate(listarry[2:]):
            vecNames.append(item[0])
            listvalues = []
            for j in item[1:]:
                listvalues.append(float(j))
            arrayValues.append(listvalues)
        return vecNames, np.array(arrayValues)

def createBarchart(listNmax, listdecay, valueArray):
    vkeys = list(valueArray.keys())
    xpos, ypos = np.meshgrid(listdecay, listNmax)
    for v in vkeys:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(xpos, ypos, valueArray[v],cmap=cm.coolwarm)
        ax.set_xlabel("decay")
        ax.set_ylabel("Nmax")
        ax.set_zlabel("avg_auc_score")
        ax.set_title(v)
        plt.show()
def getHighestAvgScore(scoreDic):
    vkeys = list(scoreDic.keys())
    gesValues = scoreDic[vkeys[0]]
    for v in vkeys[1:]:
        gesValues = gesValues + scoreDic[v]
    highestInd = np.unravel_index(np.argmax(gesValues), gesValues.shape)
    for v in vkeys:
        print(f"{v}: {scoreDic[v][highestInd]}")
    return highestInd

def mainProcess():
    listNmax, listdecay, valueArray = getFileStrukture(path)
    aucScoreMean, aucScoreLow, aucScoreHigh, avgPrecScoreMean, avgPrecScoreLow, avgPrecScoreHigh = sortToDics(valueArray)
    createBarchart(listNmax, listdecay, aucScoreMean)
    highestInd = getHighestAvgScore(aucScoreMean)
    print(f"highest auc score Nmax: {listNmax[highestInd[0]]}, Decay: {listdecay[highestInd[1]]}")

if __name__ == "__main__":
    mainProcess()
