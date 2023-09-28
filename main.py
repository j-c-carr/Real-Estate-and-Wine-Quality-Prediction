import numpy as np
import pandas as pd

from data_acquisition.data_acquisition import  fetch_housing_dataset, fetch_wine_dataset
from models.models import LinearRegression, LogisticRegression
from models.optimizers import Adam, StochasticGradientDescent
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
np.random.seed(0)

housing_df = fetch_housing_dataset()
wine_df = fetch_wine_dataset()

# 2.1 Analytic logistic regression
#X_housing_train, X_housing_test, y_housing_train, y_housing_test = train_test_split(housing_df.drop(['MEDV'], axis=1),
#                                                                                    housing_df.MEDV, test_size=0.25)

# 2.2 Logistic regression with SGD. Achieves about 91% accuracy on test set with default parameters.
one_hot_wine_classes = pd.get_dummies(wine_df['class']).to_numpy(dtype=int)
X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(wine_df.drop(['class'], axis=1),
                                                                        one_hot_wine_classes,
                                                                        test_size=0.25, stratify=wine_df['class'])
print(y_wine_train.shape)
log_r = LogisticRegression()
log_r.fit(X_wine_train, y_wine_train, optimizer_class=Adam, max_iters=1e4, verbose=True, batch_size=X_wine_train.shape[0])
y_preds = log_r.predict(X_wine_test)
y_preds_binary = (y_preds >= y_preds.max(axis=1, keepdims=True)).astype(int)
print(accuracy_score(y_wine_test, y_preds_binary))
exit()

# 2.3 Linear regression with gradient descent on dummy data
#N = 2000
#D = 13
#X = np.random.rand(N, D)
#w_star = np.random.rand(D, 1)
#y = np.dot(X, w_star)
#print(y.shape)
#lin_reg = LinearRegression(add_bias=False)
#lin_reg.fit(X, y, optimizer_class=StochasticGradientDescent, batch_size=X.shape[0], beta=0)