import json
import os
import numpy as np

train="./annotations/instances_train.json"
val="./annotations/instances_val.json"

imagaNames = os.listdir("./images")

trainList=[]
valList=[]


def getImgName(trainData):
    temp=[]
    for raw in trainData["images"]:
        temp.append(raw["file_name"])
    return temp


with open(train) as data_file:
    trainData = json.load(data_file)
    trainData["images"]=np.array(trainData["images"])
    trainFileName=getImgName(trainData)
    with open(val) as data_file:
        valData = json.load(data_file)
        valData["images"] = np.array(valData["images"])
        valFileName=getImgName(valData)
        for fileName in imagaNames:
            if fileName in trainFileName:
                trainList.append(fileName)
            if fileName in valFileName:
                valList.append(fileName)

for file in trainList:
    os.rename("images/"+file, "train/"+file)

for file in valList:
    os.rename("images/"+file, "val/"+file)

print("")