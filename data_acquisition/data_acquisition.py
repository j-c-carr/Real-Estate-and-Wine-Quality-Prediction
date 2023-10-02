import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from scipy.stats import zscore

HOUSING_FEATURE_INFO = {'CRIM': 'per capita crime rate by town',
             'ZN': 'proportion of residential land zoned for lots over 25,000 sq.ft.',
             'INDUS': 'proportion of non-retail business acres per town',
             'CHAS': 'Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)',
             'NOX': 'nitric oxides concentration (parts per 10 million)',
             'RM': 'average number of rooms per dwelling',
             'AGE': 'proportion of owner-occupied units built prior to 1940',
             'DIS': 'weighted distances to five Boston employment centres',
             'RAD': 'index of accessibility to radial highways',
             'TAX': 'full-value property-tax rate per $10,000',
             'PTRATIO': 'pupil-teacher ratio by town',
             'LSTAT': '% lower status of the population',
             'MEDV': "Median value of owner-occupied homes in $1000's"}


# fetch boston housing dataset
def fetch_housing_dataset(preprocess=True):

    df = pd.read_csv('https://raw.githubusercontent.com/j-c-carr/boston_dataset/master/boston.csv').drop(
        ['B'], axis=1)

    if preprocess:
        df.dropna(inplace=True)
        df = min_max_scale(df)

    return df


def remove_outliers(df, z_max=3):
    """Remove rows that are outliers according to z score"""
    return df[(np.abs(zscore(df)) <= z_max).all(axis=1)]


def min_max_scale(df, feature_range=(0,1)):
    scale_min, scale_max = feature_range
    df_scaled = (df - df.min()) / (df.max() - df.min())
    scaled_df = df_scaled * (scale_max - scale_min) + scale_min
    
    return pd.DataFrame(scaled_df, columns=df.columns)


# fetch wine dataset
def fetch_wine_dataset(preprocess=True):
    wine = fetch_ucirepo(id=109)
    wine_df = pd.concat([wine.data.features, wine.data.targets], axis=1)

    if preprocess:
        wine_df.dropna(inplace=True)

    return wine_df
