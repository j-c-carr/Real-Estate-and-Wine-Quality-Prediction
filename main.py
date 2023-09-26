import numpy as np
from data_processing.data_processing import  fetch_housing_dataset, fetch_wine_dataset, housing_tt_split
from data_analysis.data_analysis import dataframe_statistics
from models.models import AnalyticLinearRegression
np.random.seed(0)

housing_df = fetch_housing_dataset()
wine_df = fetch_wine_dataset()

# analyze the data
dataframe_statistics(housing_df, plot_hist=True, fig_filepath="./out/figures/housing_df_features.png")
dataframe_statistics(wine_df, plot_hist=True, fig_filepath="./out/figures/wine_df_features.png")
