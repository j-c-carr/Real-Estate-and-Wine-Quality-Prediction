import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (16, 8)

def dataframe_statistics(df, plot_hist=False, fig_filepath='./figures/test.png', nbins=50):
    print("Summary Statistics for Dataset\n", df.describe())

    if plot_hist:
        df.hist(bins=50)
        plt.savefig(fig_filepath, dpi=400)


