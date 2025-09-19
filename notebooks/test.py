import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

scriptDirectory = os.path.dirname(os.path.abspath(__file__))
csvPath = os.path.join(scriptDirectory, "..", "data", "train.csv")
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

lags = [1,2,3]
rollingAverages = [3,5,10]

for lag in lags:
    for column in inputColumns:
        dataframe[f"{column}Lag{lag}"] =  dataframe[column].shift(lag)

for rollingAverage in rollingAverages:
    for column in inputColumns:
        dataframe[f"{column}Roll{rollingAverage}"] = dataframe[column].rolling(rollingAverage).mean()

dataframe.dropna(inplace=True)

featureColumns = dataframe.drop(columns=["time","Y1","Y2"])
targetColumns = dataframe[["Y1","Y2"]]

splitPoint = int(0.75*len(dataframe))
pastfeatureRows = featureColumns.iloc[:splitPoint]
futurefeatureValue = featureColumns.iloc[splitPoint:]
firsttargetRows = targetColumns.iloc[:splitPoint]
lasttargetRows = targetColumns.iloc[splitPoint:]

modelY1 = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=50,
    subsample=0.8,
    random_state=42,
    max_features="sqrt"
)

modelY2 = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=50,
    subsample=0.8,
    random_state=42,
    max_features="sqrt"
)

modelY1.fit(pastfeatureRows, firsttargetRows["Y1"])
modelY2.fit(pastfeatureRows, firsttargetRows["Y2"])

predictedY1 = modelY1.predict(futurefeatureValue)
predictedY2 = modelY2.predict(futurefeatureValue)

scoreY1 = r2_score(lasttargetRows["Y1"], predictedY1)
scoreY2 = r2_score(lasttargetRows["Y2"], predictedY2)
averageScore = (scoreY1+scoreY2)/2
print(f"R^2 Y1: {scoreY1:.4f}")
print(f"R^2 Y2: {scoreY2:.4f}")
print(f"Average R^2 : {averageScore:.4f}")