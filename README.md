# Linear and Logistic Regression in Numpy
**Description:** This project implements three gradient-based optimization methods for linear and logistic regression in `numpy`. 
* Using the Boston Housing and Wine Datasets, we investigate how the performance of these models change with respect to several variables including the training set size, batch size, learning rate and optimization method (GD vs. SGD vs. Adam).
* Our best linear regression model achieves a Mean Squared Error of 5.396 on the Boston Housing dataset and our best logistic regression model achieves a perfect test accuracy on the Wine dataset.
* Lastly, we experiment with augmenting the Boston Housing Dataset through Gaussian and Sigmoid transformations.
  
`writeup.pdf` contains our full report.

## Installation
Before running the project, you need to set up the required environment. Follow these steps:

**1. Clone the Repository:**
```
git clone https://github.com/j-c-carr/Linear-and-Logistic-Regression.git
cd Linear-and-Logistic-Regression
```
**2. Create a Virtual Environment (Optional but Recommended):**
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
**3. Install Dependencies:**
```
pip install -r requirements.txt
```

## Usage
To use this project, follow these steps:

**1. Run Jupyter Notebooks:**
* Launch Jupyter Notebook in the project directory:
```
jupyter notebook
```
* Open the relevant Jupyter notebooks, such as:
  - `experiments.ipynb` - contains all of the linear and logistic regression experiments
  - `data_analysis/boston_analysis.ipynb`
  - `data_analysis/wine_analysis.ipynb`
  
**2. Explore the Code:**
* Review the codebase:
  - `models/models.py` - contains the linear and logistic regression models
  - `models/optimizers.py` - contains the optimizers (SGD and Adam)
 
**3. Customize and Experiment:**
* Feel free to customize parameters and experiment with the code.
* Note any additional instructions provided within the notebooks.
