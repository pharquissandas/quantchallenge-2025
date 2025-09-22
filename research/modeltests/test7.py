import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit

scriptDirectory = os.path.dirname(os.path.abspath(__file__))
trainPath = os.path.join(scriptDirectory, "..", "data", "train.csv")
testPath = os.path.join(scriptDirectory, "..", "data", "test.csv")

train = pd.read_csv(trainPath).sort_values("time").reset_index(drop=True)
test = pd.read_csv(testPath).sort_values("time").reset_index(drop=True)

features = [c for c in train.columns if c not in ("time", "Y1", "Y2")]

def create_features(df, max_lag=10, roll_windows=(5, 10, 20, 50)):
    df = df.copy()
    for c in features:
        for lag in range(1, max_lag+1):
            df[f"{c}_lag{lag}"] = df[c].shift(lag)
        for w in roll_windows:
            df[f"{c}_rollmean_{w}"] = df[c].rolling(w, min_periods=1).mean().shift(1)
            df[f"{c}_rollstd_{w}"] = df[c].rolling(w, min_periods=1).std().shift(1)
    return df

train = create_features(train)
test = create_features(test)

train["Y1_lag1"] = train["Y1"].shift(1)
train["Y2_lag1"] = train["Y2"].shift(1)

train = train.fillna(method="ffill").fillna(method="bfill")
test = test.fillna(method="ffill").fillna(method="bfill")

final_features = [c for c in train.columns if c not in ("time", "id", "Y1", "Y2")]

X = train[final_features]
y1 = train["Y1"]
y2 = train["Y2"]
X_test = test[final_features]

def objective(trial, y):
    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "num_leaves": trial.suggest_int("num_leaves", 31, 256),
        "max_depth": trial.suggest_int("max_depth", 5, 20),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
    }
    tscv = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = lgb.LGBMRegressor(**params, n_estimators=2000)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="l2", callbacks=[lgb.early_stopping(100)])
        preds = model.predict(X_val)
        scores.append(r2_score(y_val, preds))
    return np.mean(scores)

study_y1 = optuna.create_study(direction="maximize")
study_y1.optimize(lambda trial: objective(trial, y1), n_trials=30)

study_y2 = optuna.create_study(direction="maximize")
study_y2.optimize(lambda trial: objective(trial, y2), n_trials=30)

model_y1 = lgb.LGBMRegressor(**study_y1.best_params, n_estimators=2000)
model_y1.fit(X, y1)

model_y2 = lgb.LGBMRegressor(**study_y2.best_params, n_estimators=2000)
model_y2.fit(X, y2)

pred_y1 = model_y1.predict(X_test)
pred_y2 = model_y2.predict(X_test)

sub = pd.DataFrame({"id": test["id"], "Y1": pred_y1, "Y2": pred_y2})
sub.to_csv(os.path.join(scriptDirectory, "submission.csv"), index=False)