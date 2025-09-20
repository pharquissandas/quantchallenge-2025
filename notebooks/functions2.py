import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def cleanData(csvPath):
    dataframe = pd.read_csv(csvPath)

    dataframe.ffill(inplace=True)
    dataframe.fillna(dataframe.mean(), inplace=True)

    for column in dataframe.columns[1:]:
        standardDeviation = dataframe[column].std()
        mean = dataframe[column].mean()
        upperBound = mean+3*standardDeviation
        lowerBound = mean-3*standardDeviation
        dataframe[column] = dataframe[column].clip(lowerBound,upperBound)

    inputColumns = dataframe.columns[1:15]
    scaler = StandardScaler()
    dataframe[inputColumns] = scaler.fit_transform(dataframe[inputColumns])

    return dataframe, inputColumns


def featureEngineering(dataframe, inputColumns, lags=[1,2,3,5,10], rollingAverages = [3,5,10,20], momentumLags=[5,10,20]):
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
            dataframe[f"{column}momentum{momentumLag}"] = (dataframe[column] - dataframe[column].shift(momentumLag)) / (dataframe[column].shift(momentumLag) + 1e-8)

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
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    lgbModel = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8
    )

    xgbmodelY1 = clone(xgbModel)
    xgbmodelY2 = clone(xgbModel)
    lgbmodelY1 = clone(lgbModel)
    lgbmodelY2 = clone(lgbModel)

    xgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    xgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])
    lgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    lgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])

    xgbY2 = xgbmodelY2.predict(futurefeatureValue)
    lgbY2 = lgbmodelY2.predict(futurefeatureValue)
    xgbscoreY2 = r2_score(lasttargetRows["Y2"], xgbY2)
    lgbscoreY2 = r2_score(lasttargetRows["Y2"], lgbY2)

    totalR2 = xgbscoreY2+lgbscoreY2
    weightXGB = xgbscoreY2/totalR2
    weightLGB = lgbscoreY2/totalR2

    predictedY1 = (xgbmodelY1.predict(futurefeatureValue) + lgbmodelY1.predict(futurefeatureValue))/2
    predictedY2 = weightXGB*xgbY2 + weightLGB*lgbY2

    scoreY1 = r2_score(lasttargetRows["Y1"], predictedY1)
    scoreY2 = r2_score(lasttargetRows["Y2"], predictedY2)
    averageScore = (scoreY1+scoreY2)/2
    print(f"R^2 Y1: {scoreY1:.4f}")
    print(f"R^2 Y2: {scoreY2:.4f}")
    print(f"Average R^2 : {averageScore:.4f}")

    return predictedY1, predictedY2

# Main pipeline for changing models

if __name__ == "__main__":
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    csvPath = os.path.join(scriptDirectory, "..", "data", "train.csv")

    dataframe, inputColumns = cleanData(csvPath)
    dataframe = featureEngineering(dataframe, inputColumns)

    pastfeatureRows, futurefeatureValue, firsttargetRows, lasttargetRows = dataSplit(dataframe, splitPoint=0.75)

    # change model here

    modelsTogether(pastfeatureRows, futurefeatureValue, firsttargetRows, lasttargetRows)