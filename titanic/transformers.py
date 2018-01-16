#!/usr/bin/env python

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline


class Transformer:
    def __init__(self):
        self.columns = 'Sex Ticket Cabin Embarked'.split()
        self.attribute_names = 'Sex Embarked Pclass Fare'.split()

    def get_columns(self, df):
        for i in self.columns:
            df[i] = df[i].astype('category').values.codes
        return df

    def get_numericals(self, df):
        df1 = pd.DataFrame()
        for name, group in df.groupby(df.dtypes, axis=1):
            if name != 'object':
                for i in group.columns:
                    df1[i] = df[i].copy()
        return df1

    def get_x_y(self, df):
        result = (
                df
                .pipe(self.get_columns)
                .pipe(self.get_numericals)
            )

        xs = result[self.attribute_names].values

        transformer = Pipeline([
            ('imputer', Imputer(strategy='median'))
        ])
        X = transformer.fit_transform(xs)

        try:
            y = result['Survived']
            return X, y
        except:
            return X, None
