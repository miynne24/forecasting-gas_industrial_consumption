#packages needed for the scripts

import pandas as pd #package for data handling
import numpy as np #package for default arithmetic
import statsmodels.tsa as tsa #package for time series econometric models
import matplotlib.pyplot as plt #package for plotting graph
import seaborn as sns #package for plotting nicer graph
import statsmodels.api as sm #package for models and other statistical test
from statsmodels.tsa.stattools import adfuller #package for ADF test for stationarity
from arch.unitroot import PhillipsPerron as pperron #package for Phillips Perron test for stationarity
from statsmodels.tsa.stattools import kpss #package for KPSS test for stationarity
from statsmodels.tsa.stattools import acf, pacf #package for acf and pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #package to plot autocorrelation function and partial
from statsmodels.tsa.ardl import ardl_select_order #package for selecting optimal lag order of ARDL model
from statsmodels.tsa.ardl import ARDL #package for ARDL modeling
from sklearn.model_selection import train_test_split #package for splitting series
from sklearn.model_selection import TimeSeriesSplit #package for splitting series
from statsmodels.tools.eval_measures import rmse #package for RMSE calculation
from pandas.tseries.offsets import DateOffset #package to do date arithmetic
from statsmodels.tsa.arima.model import ARIMA as arima #package for ARIMA
from statsmodels.tools.eval_measures import rmse #package for RMSE calculation
from statsmodels.regression.linear_model import OLS as ols #package for ols
from typing import Type #package for ols diagnostic
from statsmodels.stats.diagnostic import het_breuschpagan #package for Breusch Pagan test for heteroscedasticity
from statsmodels.stats.diagnostic import acorr_ljungbox #package for ljungbox test for white noise
from statsmodels.stats.diagnostic import linear_harvey_collier #package for Harvey Collier test for linearity
from statsmodels.stats.diagnostic import linear_lm #package for Lagrange multiplier test for linearity
from statsmodels.stats.outliers_influence import variance_inflation_factor #package for vif to test for multicollinearity
from statsmodels.stats.stattools import jarque_bera # package for jarque bera test of normality
from statsmodels.tools.tools import maybe_unwrap_results #package for regression diagnostics graphics
from statsmodels.graphics.gofplots import ProbPlot #package for regression diagnostics graphics
from datetime import datetime #package for handling dates
from statsmodels.tsa.vector_ar.var_model import VAR #package for VAR model
from statsmodels.tsa.vector_ar.vecm import select_coint_rank #package for Johansen cointegration test
from statsmodels.tsa.vector_ar.vecm import coint_johansen #package for Johansen cointegration test
from statsmodels.tsa.vector_ar.vecm import select_order as vecm_select_order #package for selecting optimal lag length
from statsmodels.tsa.vector_ar.vecm import VECM #package for VECM model
import statistics as st #package for statistical stuff
import os #package for file operations
import openpyxl #package for excel operations
from sklearn.svm import SVR #package for Support Vector Regression
from sklearn.neighbors import KNeighborsRegressor #package for K-Nearest Neighbors regression
from sklearn.ensemble import RandomForestRegressor #package for Random Forest regression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV #package for model validation and hyperparameter tuning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score #package for model evaluation metrics
from sklearn.preprocessing import StandardScaler #package for scaling and standardizing features
from sklearn.svm import LinearSVR #package for linear Support Vector Regression
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectPercentile, chi2, f_classif, RFE, f_regression #package for feature selection techniques
from sklearn.inspection import permutation_importance #package for evaluating feature importance using permutation
from sklearn.pipeline import Pipeline #package for building machine learning pipelines
import warnings #package for managing warnings in code
from pandas.tseries.offsets import QuarterEnd #package for handling quarter-end date operations
from google.colab import drive #package for accessing Google Drive files in Google Colab
import os #package for file and directory operations
warnings.filterwarnings("ignore") #package for ignoring warnings during runtime
