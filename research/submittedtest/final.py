import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

scriptDirectory = os.getcwd()
trainPath = os.path.join(scriptDirectory, "train.csv")
testPath = os.path.join(scriptDirectory, "test.csv")

trainDataframe = pd.read_csv(trainPath)
testDataframe = pd.read_csv(testPath)

def timeFeatures(dataframe):
    dataframe = dataframe.copy()
    dataframe["timeNorm"] = dataframe["time"]/dataframe["time"].max()
    return dataframe

trainDataframe = timeFeatures(trainDataframe)
testDataframe = timeFeatures(testDataframe)

def rollingFeatures(dataframe, window=5):
    dataframe = dataframe.copy()
    for column in ["A","B","C","D"]:
        dataframe[f"{column}RollMean"] = dataframe[column].rolling(window).mean().fillna(0)
        dataframe[f"{column}RollSTD"] = dataframe[column].rolling(window).std().fillna(0)
    return dataframe

trainDataframe = rollingFeatures(trainDataframe)
testDataframe = rollingFeatures(testDataframe)

inputColumns = trainDataframe.drop(columns=["Y1","Y2"])
y1Column = trainDataframe["Y1"]
y2Column = trainDataframe["Y2"]
xTest = testDataframe.drop(columns="id")

crossValidation = TimeSeriesSplit(n_splits=5)
r2ScoreY1, r2ScoreY2 = [],[]


def modelTraining():
    return XGBRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=8,
        subsample=0.9,
        random_state=42,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        reg_alpha=0.5,
        reg_lambda=1.0,
        gamma=0.1,
        min_child_weight=3
    )

for trainIndex, valIndex in crossValidation.split(inputColumns):
    xTrain = inputColumns.iloc[trainIndex]
    xVal = inputColumns.iloc[valIndex]
    y1Train = y1Column.iloc[trainIndex]
    y1Val = y1Column.iloc[valIndex]
    y2Train = y2Column.iloc[trainIndex]
    y2Val = y2Column.iloc[valIndex]

    y1Model = modelTraining()
    y2Model = modelTraining()

    y1Model.fit(xTrain, y1Train, eval_set=[(xVal,y1Val)], verbose=False)
    y2Model.fit(xTrain, y2Train, eval_set=[(xVal,y2Val)], verbose=False)

    predictedY1 = y1Model.predict(xVal)
    predictedY2 = y2Model.predict(xVal)

    r2ScoreY1.append(r2_score(y1Val,predictedY1))
    r2ScoreY2.append(r2_score(y2Val,predictedY2))

print(f"Y1 R2: {np.mean(r2ScoreY1):.4f}")
print(f"Y2 R2: {np.mean(r2ScoreY2):.4f}")
print(f"Average R2: {(np.mean(r2ScoreY1)+np.mean(r2ScoreY2))/2:.4f}")

finalModelY1 = modelTraining()
finalModelY2 = modelTraining()

finalModelY1.fit(inputColumns, y1Column, verbose=False)
finalModelY2.fit(inputColumns, y2Column, verbose=False)

predtestY1 = finalModelY1.predict(xTest)
predtestY2 = finalModelY2.predict(xTest)

final = pd.DataFrame({
    "id":testDataframe["id"],
    "Y1":predtestY1,
    "Y2":predtestY2
})

finalFile = os.path.join(scriptDirectory, "final.csv")
final.to_csv(finalFile, index=False)

