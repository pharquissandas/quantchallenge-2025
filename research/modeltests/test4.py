import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def cleanData(csvPath):
    dataframe = pd.read_csv(csvPath)
    inputColumns = []
    for column in dataframe.columns:
        if column!="time" and column!="Y1" and column!="Y2":
            inputColumns.append(column)

    dataframe.ffill(inplace=True)
    dataframe.fillna(dataframe.mean(), inplace=True)

    for column in inputColumns + ["Y1", "Y2"]:
        standardDeviation = dataframe[column].std()
        mean = dataframe[column].mean()
        upperBound = mean+3*standardDeviation
        lowerBound = mean-3*standardDeviation
        dataframe[column] = dataframe[column].clip(lowerBound,upperBound)

    scaler = StandardScaler()
    dataframe[inputColumns] = scaler.fit_transform(dataframe[inputColumns])

    return dataframe, inputColumns

def featureEngineering(dataframe, inputColumns, lags=[1,2,3,5,10,20,30,50], rollingAverages = [3,5,10,20,50], momentumLags=[5,10,20]):
    for lag in lags:
        for column in inputColumns:
            dataframe[f"{column}Lag{lag}"] =  dataframe[column].shift(lag)
            dataframe[f"{column}Diff{lag}"] = dataframe[column] - dataframe[column].shift(lag)

    for rollingAverage in rollingAverages:
        for column in inputColumns:
            dataframe[f"{column}roll{rollingAverage}"] = dataframe[column].rolling(rollingAverage).mean()
            dataframe[f"{column}rollingSTD{rollingAverage}"] = dataframe[column].rolling(rollingAverage).std()

    for momentumLag in momentumLags:
        for column in inputColumns:
            dataframe[f"{column}momentum{momentumLag}"] = (dataframe[column] - dataframe[column].shift(momentumLag)) / (dataframe[column].shift(momentumLag)+1e-8)

    dataframe.dropna(inplace=True)
    return dataframe

def dataSplit(dataframe, splitPoint=0.75):
    featureColumns = dataframe.drop(columns=["time","Y1","Y2"])
    targetColumns = dataframe[["Y1","Y2"]]

    splitPoint = int(splitPoint*len(dataframe))
    pastfeatureRows = featureColumns.iloc[:splitPoint]
    futurefeatureValue = featureColumns.iloc[splitPoint:]
    firsttargetRows = targetColumns.iloc[:splitPoint]
    lasttargetRows = targetColumns.iloc[splitPoint:]

    return pastfeatureRows, futurefeatureValue, firsttargetRows, lasttargetRows

def modelsTogether(pastfeatureRows, futurefeatureValue, firsttargetRows, lasttargetRows):

    xgbModel = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    lgbModel = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8
    )

    catModel = CatBoostRegressor(
        iterations=400,
        learning_rate=0.04,
        depth=5,
        verbose=0,
        random_seed=42
    )

    xgbmodelY1 = clone(xgbModel)
    xgbmodelY2 = clone(xgbModel)
    lgbmodelY1 = clone(lgbModel)
    lgbmodelY2 = clone(lgbModel)
    catmodelY1 = clone(catModel)
    catmodelY2 = clone(catModel)

    xgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    xgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])
    lgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    lgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])
    catmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    catmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])

    predictedY1 = [xgbmodelY1.predict(futurefeatureValue),lgbmodelY1.predict(futurefeatureValue),catmodelY1.predict(futurefeatureValue)]
    predictedY2 = [xgbmodelY2.predict(futurefeatureValue),lgbmodelY2.predict(futurefeatureValue),catmodelY2.predict(futurefeatureValue)]

    valuesR2 = [r2_score(lasttargetRows["Y2"], prediction) for prediction in predictedY2]
    totalValues = sum(valuesR2)
    weighting = [a/totalValues for a in valuesR2]

    y1Pred = sum(prediction/3 for prediction in predictedY1)
    y2Pred = sum(weight*prediction for weight,prediction in zip(weighting, predictedY2))

    scoreY1 = r2_score(lasttargetRows["Y1"], y1Pred)
    scoreY2 = r2_score(lasttargetRows["Y2"], y2Pred)
    averageScore = (scoreY1+scoreY2)/2
    print(f"R^2 Y1: {scoreY1:.4f}")
    print(f"R^2 Y2: {scoreY2:.4f}")
    print(f"Average R^2 : {averageScore:.4f}")

    return y1Pred, y2Pred

# Main pipeline for changing models

if __name__ == "__main__":
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    csvPath = os.path.join(scriptDirectory, "..", "data", "train.csv")

    dataframe, inputColumns = cleanData(csvPath)
    dataframe = featureEngineering(dataframe, inputColumns)

    pastfeatureRows, futurefeatureValue, firsttargetRows, lasttargetRows = dataSplit(dataframe, splitPoint=0.75)

    # change model here

    modelsTogether(pastfeatureRows, futurefeatureValue, firsttargetRows, lasttargetRows)