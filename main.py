import numpy as np

import pandas as pd

from data_acquisition.data_acquisition import  fetch_housing_dataset, fetch_wine_dataset
from models.models import LinearRegression, LogisticRegression
from models.optimizers import Adam, StochasticGradientDescent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

np.random.seed(0)

housing_df = fetch_housing_dataset(preprocess=True)
wine_df = fetch_wine_dataset()


# 2.2 Logistic regression with SGD. Achieves about 91% accuracy on test set with default parameters.
one_hot_wine_classes = pd.get_dummies(wine_df['class']).to_numpy(dtype=int)
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(wine_df.drop(['class'], axis=1),
                                                                        one_hot_wine_classes,
                                                                        test_size=0.25, stratify=wine_df['class'])

log_r = LogisticRegression()
log_r.fit(X_wine_train, y_wine_train, optimizer_class=StochasticGradientDescent, max_iters=1e4, verbose=True, batch_size=X_wine_train.shape[0])
y_preds = log_r.predict(X_wine_test)
y_preds_bin = (y_preds >= y_preds.max(axis=1, keepdims=True)).astype(int)
print('sklean accuracy: ', accuracy_score(y_wine_test, y_preds_bin))

# 2.3 Linear regression with gradient descent
X_housing_train, X_housing_test, y_housing_train, y_housing_test = train_test_split(housing_df.drop(['MEDV'], axis=1).to_numpy(),
                                                                                    housing_df.MEDV.to_numpy(), test_size=0.25)

lin_reg = LinearRegression(add_bias=True)
lin_reg.fit(X_housing_train, y_housing_train, optimizer_class=StochasticGradientDescent, batch_size=X_housing_train.shape[0], beta=0)