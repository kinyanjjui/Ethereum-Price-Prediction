#!/usr/bin/env python
# coding: utf-8


#Data Pre-processing packages
import numpy as np
import pandas as pd
from datetime import datetime


#Data Visualization Libraries
import seaborn as sns
sns.set(rc={'figure.figsize': (10,6)})
custom_colors = ['#78579a','#f1b032','#8fce00']

#matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg

#colorama
from colorama import Fore, Back,Style
y_ =Fore.LIGHTYELLOW_EX
m_ =Fore.RED

#time series analysis packages
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#Facebook Prohet Packages
from fbprophet import Prophet
from fbprophet.diagnostics  import cross_validation, performance_metrics
from fbprophet.plot import add_changepoints_to_plot, plot_cross_validation_metric


data = pd.read_csv('/content/drive/MyDrive/DataSets/Ethereum Historical Data.csv')


# Dataset Overview

print(f'{m_}Total Records:{y_}{data.shape}\n')
print(f'{m_}DataTypes of data columns: \n{y_}{data.dtypes}')

data.info()
data.head(10)

#converting of the date column to datetime format and sorting the values by date
data['Date']=pd.to_datetime(data['Date'], infer_datetime_format=True, format='%y-%m-%d')
data.sort_values(by='Date',inplace=True)
data.rename(columns={'Close':'Price'}, inplace=True)
data.head()

#missing values
missed = pd.DataFrame()
missed['column'] = data.columns
missed['percent'] = [round(100*data[col].isnull().sum()/len(data), 2) for col in data.columns]
missed.sort_values('percent', ascending=False, inplace=True)
missed

## Feature Distribution

#visualizing distribution of key variables 
#opening price, 
#closing price as "Price"
#change in Eth

def triple_plot(x,title, c):
  """ 
  Gives Three Plots: Distribution, Box Plot, Violin Plot
  Arguments, x:A feature of the dataset
  title: a string
  color:  astring
  """
  fig, ax = plt.subplots(3,1, figsize=(20,10), sharex=True)
  sns.distplot(x, ax=ax[0], color=c)
  ax[0].set(xlabel=None)
  ax[0].set_title('Histogram + KDE')
  sns.boxplot(x, ax=ax[1], color=c)
  ax[1].set(xlabel=None)
  ax[1].set_title('Boxplot')
  sns.violinplot(x, ax=ax[2], color=c)
  ax[2].set(xlabel=None)
  ax[2].set_title('Violin Plot')
  fig.suptitle(title, fontsize=26)
  plt.tight_layout(pad=3.0)
  plt.show()

triple_plot(data['Price'], 'Distribution of Price', custom_colors[0])
triple_plot(data['Open'], 'Distribution of Opening Price', custom_colors[1])
triple_plot(data["High"], 'Distribution of Highest Price', custom_colors[2])
triple_plot(data['Low'], 'Distribution of Lowest Price', custom_colors[0])
triple_plot(data['Volume'], 'Distribution of Volume', custom_colors[2])


# Correlation Analysis
plt.figure(figsize=(10,10))
corr = data[data.columns[1:]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=0.3, center=0,
            square=True, linewidths=.5, annot=True,)
plt.show()

#corr

# Time Series Analysis and Prediction using Prophet
plt.figure(figsize=(20,15))
series = data[['Date','Price']]
series.set_index('Date', inplace=True)
srs = series['Price']
result = seasonal_decompose(srs, model='additive',freq=1)
result.plot()

#preparing input into Prophet
prophet_df = data[['Date','Price']]
prophet_df.rename(columns = {'Date':'ds',"Price":'y'}, inplace=True)
prophet_df.head()

#creating and fitting the prophet model with default values
prophet_basic = Prophet()
prophet_basic.fit(prophet_df[['ds','y']])

# Predicting Future Values
#Extending 1 year(365 days) into the future
future = prophet_basic.make_future_dataframe(periods=365, freq='D')
future.tail(5)

#prediction and plotting of predicted data
forecast = prophet_basic.predict(future)
pred_fig = prophet_basic.plot(forecast)
#plotting forecasted components: Trends and Seasonality
comp_fig = prophet_basic.plot_components(forecast)
# Adding of Changepoints to Prophet Prediction
#Changepoints: points of abrupt changes in the time-series trajectory
pred_fig = prophet_basic.plot(forecast)
cp_ = add_changepoints_to_plot(pred_fig.gca(), prophet_basic, forecast, cp_color ='#f1b032')
print(f'{m_}ChangePoints:\n {y_}{prophet_basic.changepoints}\n')


# Adding Multiple Regressors to the Prophet Model
prophet_df['Open'] = data['Open']
prophet_df['High'] = data['High']
prophet_df['Volume'] = data['Volume']
prophet_df['Low'] = data['Low']

#creating training and test sets
prophet_df = prophet_df.dropna()
train_X = prophet_df[:round(0.8*len(prophet_df))]
test_X = prophet_df[round(0.8*len(prophet_df)):]

pro_regressor = Prophet()
pro_regressor.add_regressor('Open')
pro_regressor.add_regressor('High')
pro_regressor.add_regressor('Low')
pro_regressor.add_regressor('Volume')

#fitting the data
pro_regressor.fit(train_X)
future_data = pro_regressor.make_future_dataframe(periods=365,freq='D')
#predicting against the test set
forecast_data = pro_regressor.predict(test_X)
pro_regressor.plot(forecast_data)

# Cross Validation and Running Performance Tests on the Model
#Using the first 700 days as training data; as the first cutoff 
#cross validating using half-year sets for every alternate 90-day period
df_cv = cross_validation(pro_regressor,initial='700 days', period='90 days',
                         horizon='180 days')

df_cv.head(10)

pm =performance_metrics(df_cv, rolling_window=0.1)
display(pm.head(), pm.tail())

#plotting mape to see if it falls below the .15 threshold
fig = plot_cross_validation_metric(df_cv, metric='mape', rolling_window=0.1)
plt.show()
