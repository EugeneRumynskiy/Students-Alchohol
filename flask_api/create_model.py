import os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
import dill as pickle

import warnings
warnings.filterwarnings("ignore")


def build_and_train():
    # loading data
    df = pd.read_csv("../data/student-mat.csv", ";")

    target_col = "G3"
    x, y = df.drop(target_col, axis=1), df[target_col]

    # making pipeline
    regressor = RandomForestRegressor()
    preproc = PreProcessing()
    pipe = Pipeline(steps=[('preproc', preproc), ('regressor', regressor)])

    # setting up grid search
    #param_grid = {"regressor__max_iter": [100, 200, 1000],
    #              "regressor__alpha": [0.0001, 0.001, 0.01, 0.1, 1],
    #              "regressor__l1_ratio": np.arange(0.1, 0.5, 0.1)}
    param_grid = {"regressor__n_estimators": [20, 50, 100, 200],
                  "regressor__max_depth": [2, 5, 7, 10, 15],
                  "regressor__criterion": ["mae", "mse"]}

    # running grid search
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=3)
    grid.fit(x, y)

    return grid


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """

    def __init__(self):
        pass

    def transform(self, df_to_process):
        """Regular transform() that is a help for training, validation & testing datasets
           (NOTE: The operations performed here are the ones that we did prior to this cell)
        """

        def process_numerical(df):
            numerical = ["age", 'Medu', "Fedu", "traveltime", "studytime", "failures",
                         "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]

            # cutting off rare values
            df.loc[df["age"] > 19, 'age'] = 19
            df.loc[(df["Dalc"] > 3).index, 'Dalc'] = 3
            # We are still gettig warn of chained assignment here, though It's false positive and safe

            return df[numerical]

        # not used
        def process_binary(df):
            categorical = ["address", "famsize", "Pstatus", "schoolsup", "famsup", "paid",
                           "activities", "nursery", "higher", "internet", "romantic",
                           'school', 'sex', 'Mjob', 'Fjob', 'guardian', 'reason']

            # binary = list(filter(lambda col: df[col].value_counts().shape[0] == 2, categorical))
            # Это жесткая ошибка. При неравномерном распределении по признакам, при кроссвалидации
            # можно отобрать разные фичи. То есть в одном фолде значения будут бинарные, в другом нет. Всё, беда.
            # Пропишем руками.
            binary = ['address', 'famsize', 'Pstatus', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
                      'higher', 'internet', 'romantic', 'school', 'sex']

            # encoding binary variables
            schoolsup_values = {'no': 0, 'yes': 1}
            famsup_values = {'no': 0, 'yes': 1}
            paid_values = {'no': 0, 'yes': 1}
            activities_values = {'no': 0, 'yes': 1}
            nursery_values = {'no': 0, 'yes': 1}
            higher_values = {'no': 0, 'yes': 1}
            internet_values = {'no': 0, 'yes': 1}
            romantic_values = {'no': 0, 'yes': 1}

            sex_values = {'F': 0, 'M': 1}  # male\female
            address_values = {'U': 0, 'R': 1}  # urban\rural
            famsize_values = {'GT3': 1, 'LE3': 0}  # le3 == (<= 3)
            Pstatus_values = {'T': 0, 'A': 1}  # together\apart
            school_values = {'GP': 0, 'MS': 1}  # school name

            # cutting off categorical features with 3+ values
            tmp = df[binary].replace({'address': address_values, 'famsize': famsize_values, 'Pstatus': Pstatus_values,
                                      'schoolsup': schoolsup_values, 'famsup': famsup_values, 'paid': paid_values,
                                      'activities': activities_values, 'nursery': nursery_values,
                                      'higher': higher_values,
                                      'internet': internet_values, 'romantic': romantic_values, 'school': school_values,
                                      'sex': sex_values})

            # chosing features from discovery notebook
            return tmp[["address", "schoolsup", "higher", "internet", "romantic"]]

        def make_transform(df):
            # turned out that categorical features are not that important
            # binary_array = process_binary(df).as_matrix()
        
            numerical_array = process_numerical(df).as_matrix()
            numerical_array = self.numerical_pipe_.transform(numerical_array)

            # turned out that categorical features are not that important
            # return np.hstack((numerical_array, binary_array))

            return numerical_array

        return make_transform(df_to_process)

    def fit(self, df, y=None, **fit_params):
        """Fitting the Training dataset & calculating the required values from train
           e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in
                transformation of X_test
        """
        numerical = ["age", 'Medu', "Fedu", "traveltime", "studytime", "failures", 
               "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences", "G1", "G2"]

        # numerical_pipe = make_pipeline(PCA(n_components=10, svd_solver='full'), StandardScaler())
        # turned out that STDscaling is the best option here
        numerical_pipe = make_pipeline(StandardScaler())
        self.numerical_pipe_ = numerical_pipe.fit(df[numerical], y)
        
        return self


if __name__ == '__main__':
    model = build_and_train()

    filename = 'model_v1.pk'
    with open('models/' + filename, 'wb') as file:
        pickle.dump(model, file)
