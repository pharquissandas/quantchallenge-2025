import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def cleanData(csvPath, scaler=None, train=True):
    dataframe = pd.read_csv(csvPath)
    if train:
        exclude = ["time","Y1","Y2"]
    else:
        exclude = ["id","time"]

    inputColumns = []
    for column in dataframe.columns:
        if column not in exclude:
            inputColumns.append(column)

    dataframe.ffill(inplace=True)
    dataframe.fillna(dataframe.mean(), inplace=True)

    for column in inputColumns:
        standardDeviation = dataframe[column].std()
        mean = dataframe[column].mean()
        upperBound = mean+3*standardDeviation
        lowerBound = mean-3*standardDeviation
        dataframe[column] = dataframe[column].clip(lowerBound,upperBound)

    if scaler is None:
        scaler = StandardScaler()
        dataframe[inputColumns] = scaler.fit_transform(dataframe[inputColumns])
    else:
        dataframe[inputColumns] = scaler.transform(dataframe[inputColumns])

    return dataframe,inputColumns,scaler

def featureEngineering(dataframe, inputColumns, lagFeatures=[1,2,3,5,10,20,30], rollingAverages = [3,5,10,20,50], momentumLags=[5,10,20]):
    for lagFeature in lagFeatures:
        for column in inputColumns:
            dataframe[f"{column}Lag{lagFeature}"] =  dataframe[column].shift(lagFeature)
            dataframe[f"{column}Diff{lagFeature}"] = dataframe[column] - dataframe[column].shift(lagFeature)

    for rollingAverage in rollingAverages:
        for column in inputColumns:
            dataframe[f"{column}roll{rollingAverage}"] = dataframe[column].rolling(rollingAverage).mean()
            dataframe[f"{column}rollingSTD{rollingAverage}"] = dataframe[column].rolling(rollingAverage).std()

    for momentumLag in momentumLags:
        for column in inputColumns:
            dataframe[f"{column}momentum{momentumLag}"] = (dataframe[column] - dataframe[column].shift(momentumLag)) / (dataframe[column].shift(momentumLag)+1e-8)
    return dataframe

def dataSplit(dataframe, splitPoint=0.75):
    featureColumns = dataframe.drop(columns=["time","Y1","Y2"])
    targetColumns = dataframe[["Y1","Y2"]]

    splitPoint = int(splitPoint*len(dataframe))
    pastfeatureRows = featureColumns.iloc[:splitPoint]
    futurefeatureValue = featureColumns.iloc[splitPoint:]
    firsttargetRows = targetColumns.iloc[:splitPoint]
    lasttargetRows = targetColumns.iloc[splitPoint:]

    return pastfeatureRows,futurefeatureValue,firsttargetRows,lasttargetRows

def modelsTogether(pastfeatureRows, futurefeatureValue, firsttargetRows, lasttargetRows):

    lgbModel = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8
    )

    xgbModel = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    lgbmodelY1 = clone(lgbModel)
    lgbmodelY2 = clone(lgbModel)
    xgbmodelY1 = clone(xgbModel)
    xgbmodelY2 = clone(xgbModel)

    lgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    lgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])
    xgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    xgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])


    lgbY1 = lgbmodelY1.predict(futurefeatureValue)
    xgbY1 = xgbmodelY1.predict(futurefeatureValue)
    lgbY2 = lgbmodelY2.predict(futurefeatureValue)
    xgbY2 = xgbmodelY2.predict(futurefeatureValue)

    lgbScore = r2_score(lasttargetRows["Y2"], lgbY2)
    xgbScore = r2_score(lasttargetRows["Y2"], xgbY2)
    sum = lgbScore+xgbScore
    lgbWeight=lgbScore/sum
    xgbWeight=xgbScore/sum

    predictedValueY1 = (lgbY1+xgbY1)/2
    predictedValueY2 = (lgbWeight*lgbScore)+(xgbWeight*xgbScore)

    scoreY1 = r2_score(lasttargetRows["Y1"], predictedValueY1)
    scoreY2 = r2_score(lasttargetRows["Y2"], predictedValueY2)
    averageScore = (scoreY1+scoreY2)/2
    print(f"R^2 Y1: {scoreY1:.4f}")
    print(f"R^2 Y2: {scoreY2:.4f}")
    print(f"Average R^2 : {averageScore:.4f}")

    return lgbmodelY1,lgbmodelY2,xgbmodelY1,xgbmodelY2,lgbWeight,xgbWeight

def submit(trainDataframe,testDataframe,inputColumns,models,finalPath="final.csv"):
    lgbmodelY1,lgbmodelY2,xgbmodelY1,xgbmodelY2,lgbWeight,xgbWeight = models

    maxLag = 50
    fullDataframe = pd.concat([trainDataframe.tail(maxLag), testDataframe], axis=0).reset_index(drop=True)
    fullDataframe = featureEngineering(fullDataframe,inputColumns).dropna().reset_index(drop=True)

    testColumns = fullDataframe.iloc[maxLag:].drop(columns=["time","id"],errors="ignore")
    predictedY1 = (lgbmodelY1.predict(testColumns) + xgbmodelY1.predict(testColumns))/2
    predictedY2 = (lgbWeight*lgbmodelY2.predict(testColumns))+(xgbWeight*xgbmodelY2.predict(testColumns))

    final = pd.DataFrame({"id":testDataframe["id"],"Y1":predictedY1,"Y2":predictedY2})
    final.to_csv(finalPath, index=False)
    print(f"Saved to {finalPath}")

if __name__ == "__main__":
    trainPath = os.path.join("..", "data", "train.csv")
    testPath  = os.path.join("..", "data", "test.csv")

    trainDataframe, inputColumns, scaler = cleanData(trainPath, train=True)
    testDataframe, _, _ = cleanData(testPath, scaler=scaler, train=False)

    trainDataframe = featureEngineering(trainDataframe, inputColumns)
    trainDataframe = trainDataframe.dropna().reset_index(drop=True)

    pastfeatureRows,futurefeatureValue,firsttargetRows,lasttargetRows = dataSplit(trainDataframe)
    models = modelsTogether(pastfeatureRows,futurefeatureValue,firsttargetRows,lasttargetRows)

    submit(trainDataframe,testDataframe,inputColumns,models,finalPath="final.csv")