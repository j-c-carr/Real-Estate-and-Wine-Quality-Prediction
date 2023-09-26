import numpy as np
from data_processing.data_processing import  fetch_housing_dataset, fetch_wine_dataset, housing_tt_split
from data_analysis.data_analysis import dataframe_statistics
from models.models import AnalyticLinearRegression
np.random.seed(0)

housing_df = fetch_housing_dataset()
wine_df = fetch_wine_dataset()

# analyze the data
# dataframe_statistics(housing_df, plot_hist=True, fig_filepath="./out/figures/housing_df_features.png")
# dataframe_statistics(wine_df, plot_hist=True, fig_filepath="./out/figures/wine_df_features.png")

# 2.1 Analytic logistic regression
X_housing_train, X_housing_test, y_housing_train, y_housing_test = housing_tt_split(housing_df, test_size=0.25)

analytic_lin_reg = AnalyticLinearRegression()
analytic_lin_reg.fit(X_housing_train, y_housing_train)
preds = analytic_lin_reg.predict(X_housing_test)

ls_loss = 0.5 * np.dot(np.transpose(y_housing_test - preds), y_housing_test - preds)
print('Least squares loss for analytic linear regression: ', ls_loss)