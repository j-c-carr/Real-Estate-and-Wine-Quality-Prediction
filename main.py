import numpy as np
from data_processing.data_processing import  fetch_housing_dataset, fetch_wine_dataset, housing_tt_split, wine_tt_split
from data_analysis.data_analysis import dataframe_statistics
np.random.seed(0)

housing_df = fetch_housing_dataset()
wine_df = fetch_wine_dataset()

# 2.1 Analytic logistic regression
#X_housing_train, X_housing_test, y_housing_train, y_housing_test = housing_tt_split(housing_df, test_size=0.25)

# 2.2 Logistic regression with SGD. Achieves about 91% accuracy on test set with default parameters.
X_wine_train, X_wine_test, y_wine_train, y_wine_test = wine_tt_split(wine_df, test_size=0.25)
log_r = LogisticRegression()
log_r.fit(X_wine_train, y_wine_train, verbose=True, batch_size=X_wine_train.shape[0])
y_preds = log_r.predict(X_wine_test)
y_preds_binary = (y_preds >= y_preds.max(axis=1, keepdims=True)).astype(int)
print(accuracy_score(y_wine_test, y_preds_binary))

# 2.3 Linear regression with gradient descent on dummy data
#N = 2000
#D = 13
#X = np.random.rand(N, D)
#w_star = np.random.rand(D, 1)
#y = np.dot(X, w_star)
#print(y.shape)
#
#lin_reg = LinearRegression(add_bias=False)
#lin_reg.fit(X, y)
