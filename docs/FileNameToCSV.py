import csv
import os

pathSlideTable = 
pathFolder = 


def FolderToList(path):
    return os.listdir(path)

def listToCSV(path, List):
    with open(path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in List:
            writer.writerow(["", i])


if __name__ == "__main__":
    listFiles = FolderToList(pathFolder)
    print(listFiles)
    listToCSV(pathSlideTable, listFiles)


