import numpy as np

import pandas as pd

from data_acquisition.data_acquisition import  fetch_housing_dataset, fetch_wine_dataset
from models.models import LinearRegression, LogisticRegression
from models.optimizers import GradientDescent, StochasticGradientDescent, Adam
from utils.metrics import mse
from utils.data_preparation import custom_train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
np.random.seed(0)



# 2.2 Logistic regression with SGD. Achieves about 91% accuracy on test set with default parameters.
wine_df = fetch_wine_dataset()
one_hot_wine_classes = pd.get_dummies(wine_df['class']).to_numpy(dtype=int)
X_wine_train, X_wine_test, y_wine_train, y_wine_test = custom_train_test_split(wine_df.drop(['class'], axis=1).to_numpy(),
                                                                        one_hot_wine_classes,
                                                                        test_size=0.20, stratify=wine_df['class'])

log_r = LogisticRegression()
log_r.fit(X_wine_train, y_wine_train, optimizer_class=GradientDescent, max_iters=4e4, learning_rate=0.05, verbose=False)
y_preds = log_r.predict(X_wine_test)

batch_size = 8
gd_accuracy = accuracy_score(y_wine_test, y_preds)
gd_precision = precision_score(y_wine_test, y_preds, average='weighted')
gd_recall = recall_score(y_wine_test, y_preds, average='weighted')
gd_f1_score = f1_score(y_wine_test, y_preds, average='weighted')
log_r.fit(X_wine_train, y_wine_train, optimizer_class=StochasticGradientDescent, max_iters=4e4, learning_rate=0.05, verbose=False, batch_size=batch_size)
y_preds = log_r.predict(X_wine_test)

sgd_accuracy = accuracy_score(y_wine_test, y_preds)
sgd_precision = precision_score(y_wine_test, y_preds, average='weighted')
sgd_recall = recall_score(y_wine_test, y_preds, average='weighted')
sgd_f1_score = f1_score(y_wine_test, y_preds, average='weighted')

print(f'Accuracy\n\tGD: {gd_accuracy}, SGD with mini-batch of size {batch_size}: {sgd_accuracy}')
print(f'Precision\n\tGD: {gd_precision}, SGD with mini-batch of size {batch_size}: {sgd_precision}')
print(f'Recall\n\tGD: {gd_recall}, SGD with mini-batch of size {batch_size}: {sgd_recall}')
print(f'F1 score\n\tGD: {gd_f1_score}, SGD with mini-batch of size {batch_size}: {sgd_f1_score}')

# exit()
# 2.3 Linear regression with gradient descent
housing_df = fetch_housing_dataset(preprocess=True)
X_housing_train, X_housing_test, y_housing_train, y_housing_test = custom_train_test_split(housing_df.drop(['MEDV'], axis=1).to_numpy(),
                                                                                    housing_df.MEDV.to_numpy().reshape(-1,1), test_size=0.20)
lin_reg = LinearRegression(add_bias=True)
w_star = lin_reg.fit(X_housing_train, y_housing_train, analytic_fit=True)
lin_reg.fit(X_housing_train, y_housing_train, optimizer_class=StochasticGradientDescent, batch_size=X_housing_train.shape[0], record_history=True, beta=0)
print('l2 norm between w and w_star: ', np.linalg.norm(w_star - lin_reg.w))
y_preds_with_grad = lin_reg.predict(X_housing_test)
y_preds_with_analytic_fit = lin_reg.predict(X_housing_test)

ms_error = mse(y_housing_test, y_preds_with_grad)
min_ms_error = mse(y_housing_test, lin_reg.predict(X_housing_test))

print('ms_error from gradient descent: ', ms_error)
print('smallest possible ms error: ', min_ms_error)