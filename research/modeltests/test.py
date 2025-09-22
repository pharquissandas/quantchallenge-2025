import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

def cleanData(csvPath, train=True, scaler=None):
    dataframe = pd.read_csv(csvPath)

    inputColumns = [column for column in dataframe.columns if column not in ["time","Y1","Y2","id"]]

    dataframe.ffill(inplace=True)
    dataframe.fillna(dataframe.mean(), inplace=True)

    clippingColumns = inputColumns.copy()
    if train:
        clippingColumns = clippingColumns + ["Y1","Y2"]

    for column in clippingColumns:
        standardDeviation = dataframe[column].std()
        mean = dataframe[column].mean()
        upperBound = mean+3*standardDeviation
        lowerBound = mean-3*standardDeviation
        dataframe[column] = dataframe[column].clip(lowerBound,upperBound)

    if train:
        scaler = StandardScaler()
        dataframe[inputColumns] = scaler.fit_transform(dataframe[inputColumns])
        return dataframe,inputColumns,scaler
    else:
        if scaler is None:
            raise ValueError("Scaler should be there when train=False")
        else:
            return dataframe, inputColumns

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
    
    dataframe.dropna(inplace=True)
    return dataframe

def modelTraining(pastfeatureRows, firsttargetRows):
    lgbModel = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8
    )

    xgbModel = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8,
        tree_method="hist"
    )

    lgbmodelY1 = clone(lgbModel)
    lgbmodelY2 = clone(lgbModel)
    xgbmodelY1 = clone(xgbModel)
    xgbmodelY2 = clone(xgbModel)

    xgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"]) 
    xgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])
    lgbmodelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
    lgbmodelY2.fit(pastfeatureRows, firsttargetRows["Y2"])
    
    return xgbmodelY1, lgbmodelY1, xgbmodelY2, lgbmodelY2

def weightedTogether(models, A):
    xgbmodelY1, lgbmodelY1, xgbmodelY2, lgbmodelY2 = models

    predsY1 = [xgbmodelY1.predict(A), lgbmodelY1.predict(A)]
    predsY2 = [xgbmodelY2.predict(A), lgbmodelY2.predict(A)]

    weightedScores = [prediction.var() for prediction in predsY2]
    weight = sum(weightedScores) if sum(weightedScores) > 0 else 2
    weighting = [i/weight for i in weightedScores]

    finalY1 = sum(prediction/2 for prediction in predsY1)
    finalY2 = sum(a*prediction for a, prediction in zip(weighting, predsY2))

    return finalY1, finalY2

if __name__ == "__main__":
    scriptDirectory = os.path.dirname(os.path.abspath(__file__))
    trainPath = os.path.join(scriptDirectory, "..", "data", "train.csv")
    testPath = os.path.join(scriptDirectory, "..", "data", "test.csv")
    finalPath = os.path.join(scriptDirectory, "final.csv")

    trainDataframe, inputColumns, scaler = cleanData(trainPath, train=True)
    trainDataframe = featureEngineering(trainDataframe, inputColumns)

    pastfeatureRows = trainDataframe.drop(columns=["time","Y1","Y2"])
    firsttargetRows = trainDataframe[["Y1","Y2"]]

    models = modelTraining(pastfeatureRows, firsttargetRows)

    testDataframe, inputtestColumns = cleanData(testPath, train=False, scaler=scaler)
    testDataframe[inputtestColumns] = scaler.transform(testDataframe[inputtestColumns])
    testDataframe = featureEngineering(testDataframe,inputtestColumns)
    formattedTest = testDataframe.drop(columns=["id","time"])

    predY1, predY2 = weightedTogether(models, formattedTest)

    final = pd.DataFrame({
        "id":testDataframe["id"].values,
        "Y1":predY1,
        "Y2":predY2
    })

    final.to_csv(finalPath, index=False)
    print(f"Saved to {finalPath}")