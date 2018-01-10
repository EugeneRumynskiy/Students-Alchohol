import os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
import dill as pickle

import warnings
warnings.filterwarnings("ignore")


def build_and_train():
    # loading data
    df = pd.read_csv("../data/student-mat.csv", ";")
    numeric_cols = ["age", 'Medu', "Fedu", "traveltime", "studytime", "failures",
                    "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]

    x, y = df[numeric_cols], df["G3"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.1, random_state=42)

    # making pipeline
    pipe = make_pipeline(RandomForestRegressor())

    # setting up grid search
    param_grid = {"randomforestregressor__n_estimators": [10, 20, 30],
                  "randomforestregressor__max_depth": [None, 6, 8, 10],
                  "randomforestregressor__max_leaf_nodes": [None, 5, 10, 20],
                  "randomforestregressor__min_impurity_split": [0.1, 0.2, 0.3]}

    # running grid search
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    grid.fit(x_train, y_train)

    return grid


if __name__ == '__main__':
    model = build_and_train()

    filename = 'model_v1.pk'
    with open('models/' + filename, 'wb') as file:
        pickle.dump(model, file)
