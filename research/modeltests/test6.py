import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

scriptDirectory = os.getcwd()
trainPath = os.path.join(scriptDirectory, "train.csv")
testPath = os.path.join(scriptDirectory, "test.csv")

trainDataframe = pd.read_csv(trainPath)
testDataframe = pd.read_csv(testPath)

inputColumns = trainDataframe.drop(columns=["Y1","Y2"])
y1Column = trainDataframe["Y1"]
y2Column = trainDataframe["Y2"]

xTest = testDataframe.drop(columns="id")

xTrain, xVal, y1Train, y1Val = train_test_split(
    inputColumns, y1Column, test_size=0.1, shuffle=False
)

_, _, y2Train, y2Val = train_test_split(
    inputColumns, y2Column, test_size=0.1, shuffle=False
)

def modelTraining(xTrain, yTrain, xVal, yVal):
    xgbModel = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        random_state=42,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1
    )

    xgbModel.fit(
        xTrain, yTrain, eval_set=[(xVal, yVal)], verbose=False
    )
    return xgbModel

y1Model = modelTraining(xTrain, y1Train, xVal, y1Val)
y2Model = modelTraining(xTrain, y2Train, xVal, y2Val)

predictedY1 = y1Model.predict(xTest)
predictedY2 = y2Model.predict(xTest)

final = pd.DataFrame({
    "id":testDataframe["id"],
    "Y1":predictedY1,
    "Y2":predictedY2
})

finalFile = os.path.join(scriptDirectory, "final.csv")
final.to_csv(finalFile, index=False)

