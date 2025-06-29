'''
Intro to time series and stationarity 
Motivation 
Time series are everywhere 

* Science 
* Technology
* Business 
* Finance
* Policy 

ARIMA models are one of the goto time series tools 
You will learn:
- Structure of ARIMA models 
- How to fit ARIMA model
- How to optimize the model 
- How to make forecasts 
- How to calculate uncertainty in predictions

import pandas as pd
import matplotlib as plt

df = pd.read_csv('time_series.csv', index_col='date', parse_dates=True)
fig, ax = plt.subplots()
df.plot(ax=ax)
plt.show()

Trends: Positive or negative slope
Seasonality: a seasonal time series has patterns that repeat at regular intervals, 
for example high sales every weekend 
Cyclicality: there is a repeating patter but no fixed period 
White noiise: a series of measurments, where each value is uncorrelated 
with previous values. The series values are not dependent on the values that 
came before  

Stationarity: the series has zero trend, it isn't growing or shrinking. The 
variance is constant the average distance of the data points from the zero
line isn't changing and the autocorrelation is constant - how each value in a
time series is related to its neighbors stays the same

Train-test split
# Train data - all data up to the end of 2018
df_train = df.loc[:'2018']

# Test data - all data from 2019 onwards
df_test = df.loc['2019':]
'''

###Exploration
# Import modules
import matplotlib.pyplot as plt
import pandas as pd

# Load in the time series
candy = pd.read_csv('candy_production.csv', 
            index_col='date',
            parse_dates=True)

# Plot and show the time series on axis ax
fig, ax = plt.subplots()
candy.plot(ax=ax)
plt.show()

###Train-test splits
# Split the data into a train and test set
candy_train = candy.loc[:'2006']
candy_test = candy.loc['2007':]

# Create an axis
fig, ax = plt.subplots()

# Plot the train and test sets on the axis ax
candy_train.plot(ax=ax)
candy_test.plot(ax=ax)
plt.show()

'''
Making time series stationary
Statisticsal test for stationarity
Making a dataset stationary

The most common test for identifying whether a time series is non-stationary
is the augmented Dicky-Fuller test
-Test for trend non-stationarity
-Null hypothesis is time series is non-stationary

from statsmodels.tsa.stattools import adfuller

results = adfuller(df['close'])
print(results)

Output: (-1.34, 0.60, 23, 1235, {'1%': -3.435, '5%': -2.913, '10%': -2.568}, 10782.87)
The zeroth element is the test statisitc, in this case it is -1. 
The next item in the results tuple, is the test p-value
The last item is a dictionary, this stores the critical values of a test statistic
which equates to different p-values.

0th element is test statistic: (-1.34)
* More negative means more likely to be stationary
1st element is p-value: (0.60)
* If p-value is small -> reject null hypothesis. Reject non-stationary, If the 
p-value is less than 0.05 then we reject the null hypothesis. In this case we reject 
non-stationary
*4th element is a critical test statistic

A very common way to make a time series stationary is to take its difference

Difference: delta y_t = y_t - y_t-1 

df_stationary = df.diff()

Examples of other transforms 
-Take the log 
* np.log(df)
-Take the square root 
* np.sqrt(df)
-Take the proportional change
* df.shift(1)/df
'''

###Augmented Dicky-Fuller 
# Import augmented dicky-fuller test function
from statsmodels.tsa.stattools import adfuller

# Run test
result = adfuller(earthquake['earthquakes_per_year'])

# Print test statistic
print(result[0])

# Print p-value
print(result[1])

# Print critical values
print(result[4]) 

###Taking the difference 
# Calculate the first difference of the time series
city_stationary = city.diff().dropna()

# Run ADF test on the differenced time series
result = adfuller(city_stationary['city_population'])

# Plot the differenced time series
fig, ax = plt.subplots()
city_stationary.plot(ax=ax)
plt.show()

# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])

###

# Calculate the second difference of the time series
city_stationary = city.diff().diff().dropna()

# Run ADF test on the differenced time series
result = adfuller(city_stationary['city_population'])

# Plot the differenced time series
fig, ax = plt.subplots()
city_stationary.plot(ax=ax)
plt.show()

# Print the test statistic and the p-value
print('ADF Statistic:', result[0])
print('p-value:', result[1])


### Other transforms
'''
A classic way of transforming stock times series is the log-return of the series. This 
is calculated as follows: log_return(y_t) = log(y_t/y_t-1)

you can do this by the following substitution:

- y_t -> amazon
- y_t-1 -> amazon.shift(1)
- log() -> np.log()
'''

# Calculate the first difference and drop the nans
amazon_diff = amazon.diff()
amazon_diff = amazon_diff.dropna()

# Run test and print
result_diff = adfuller(amazon_diff['close'])
print(result_diff)

# Calculate log-return and drop nans
amazon_log = np.log(amazon/amazon.shift(1))
amazon_log = amazon_log.dropna()

# Run test and print
result_log = adfuller(amazon_log['close'])
print(result_log)

'''
Intro to AR, MA and ARMA models

*******
In an autoregressive model we regress the values of the time series against previous values 
of this same time series. 

Here is the equation for a simple AR(1) model: 

AR(1) model :       y_t = a_1 * y_t-1 + epselon_t

There is a shock term, white noise, epselon_t, meaning each shock is random and not related to 
the other shocks in the series 

a_1 is the autoregressive coefficient at lag 1 (slope of the line)

the dependent variable is y_t 

the independent variable is y_t-1

The order of a model ie. AR(1) 1 being the order is the number of time lags 
used. 

An AR(2) has two autoregressive coefficients, and has two independent variables, the series at 
lag one and the series at lag two. 

More generally we use p to mean the order of the AR model. 

************
In a Moving Average model we regress the values of a timeseries against the previous
shock values of this same time series. 

MA(1) model:    y_t = m_1 * epselon_t_1 + epselon_t

m_t times the values of the shock at the previous step; plus a shock term for 
the current time step. This is a first order MA model. 


An ARMA model is a combination of the AR and MA models. The time series is regressed on the previous 
values and the previous shock terms. 

ARMA(1,1) model:    y_t = a_1 * y_t-1 + m_1 * epselon_t-1 + epselon_t

More generally we use ARMA(p, q)
* p is order of AR part - tells us the order of the autoregressive part  
* q is order of MA part - tells us the order of the moving average part 

Using the statsmodels package, we can both fit ARMA models and create ARMA 
data 

Example:        y_t = 0.5 * y_t-1 + 0.2_epselon_t-1 + epselon_t
statsmodels.tsa.arima_process import arma_generate_sample

ar_coefs = [1, -0.5]
ma_coefs = [1, 0.2]
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)


both coefficients start with 1, this is for the zeroth lag term and we will 
always set this to one. 

When generating ARMA data the list of coefficients is as follows 
- The list ar_coefs [1, -a_1, -a_2, ..., -a_p]
- The list ma_coefs [1, m_1, m_2, ..., m_q]
'''

# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(1)

# Set coefficients
ar_coefs = [1]
ma_coefs = [1, -0.7]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()


####
# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(2)

# Set coefficients
ar_coefs = [1, -0.3, -0.2]
ma_coefs = [1]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()

###

# Import data generation function and set random seed
from statsmodels.tsa.arima_process import arma_generate_sample
np.random.seed(3)

# Set coefficients
ar_coefs = [1, 0.2]
ma_coefs = [1, 0.3, 0.4]

# Generate data
y = arma_generate_sample(ar_coefs, ma_coefs, nsample=100, scale=0.5)

plt.plot(y)
plt.ylabel(r'$y_t$')
plt.xlabel(r'$t$')
plt.show()


###Fitting Prelude 
# Import the ARMA model
from statsmodels.tsa.arima_model import ARMA

# Instantiate the model
model = ARMA(y, order=(1,1))

# Fit the model
results = model.fit()

'''
Fitting time series models 

from statsmodels.tsa.arima_model import ARMA
model = ARMA(timeseries, order=(p, q)) # data can be pd dataframe, pd series or np array, p is the number of autoregressive lags and q is the number of moving average lags

#Creating AR and MA models 
ar_model = ARMA(timeseries, order=(p, 0))
ma_model = ARMA(timeseries, order=(0, q))

#Fitting the model and fit summary
model = ARMA(timeseries, order=(2,1))
results = model.fits()
print(resullts.summary())

Inroduction to ARMAX models 
* Exogenous ARMA
* Use external variables as well as time series 
* ARMAX = ARMA + regression 

ARMAX equation 
ARMA(1,1) model : 
                    y_t = a_1 * y_t-1 + m_1 * epselon_t-1 + epselon_t
                    
ARMAX(1,1) model : 
                    y_t = x_1 * z_t + a_1 * y_t-1 + m_1 * epselon_t-1 + epselon_t
                    
Fitting ARMAX
# Instantiate the model 
model = ARMA(df['productivity'], order=(2, 1), exog=df['houors_sleep'])

# Fit the model 
results = model.fit()

'''

###Fitting AR and MA models 
# Instantiate the model
model = ARMA(sample['timeseries_1'], order=(2,0))

# Fit the model
results = model.fit()

# Print summary
print(results.summary())

###

# Instantiate the model
model = ARMA(sample['timeseries_2'], order=(0,3))

# Fit the model
results = model.fit()

# Print summary
print(results.summary())

###Fittng the ARMA model 
# Instantiate the model
model = ARMA(earthquake, order=(3,1))

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())

###Fitting ARMAX model 
# Instantiate the model
model = ARMA(hospital['wait_times_hrs'], order=(2, 1), exog=hospital['nurse_count'])

# Fit the model
results = model.fit()

# Print model fit summary
print(results.summary())

'''
Forecasting 
Take a AR(1) model

y_t = a_1 * y_t-1 + espelon_t

Predict next value

y_t = 0.6 * 10 + epselon_t

y_t = 6.0 + epselon_t

Uncertianty on predictions

5.0 < y_t < 7.0

Statsmodels SARIMAX class 

from statsmodels.tsa.statespace.sarimax import SARIMAX

# Just an ARMA(p, q) model
model = SARIMAX(df, order(p, 0, q), trend='c') SARIMAX model order=(p,0,q) same as ARMA model order=(p, q)

# Make predictions for last 25 values 
results = model.fit()

# Make in-sample predictions
forecast = results.get_prediction(start=-25)

# forecast mean 
mean_forecast = forecast.predicted_mean

# Get confidence intervals of forecasts
confidence_intervals = forecast.conf_int()

plt.figure()

# Plot predictions
plt.plot(dates,
        mean_forecast.values,
        color='red',
        label='forecast')

# Shade uncertainty area
plt.fill_between(dates, lower_limits, upper_limits, color='pink')

plt.show()

Making dynamic predicions
results = model.fit()
forecast = results.get_prediction(start=-25, dynamic=True)

# forecast mean
mean_forecast = forecast.predicted_mean

# Get confidence intervals of forecasts
confidence_intervals = forecast.conf_int()

Forecastin out of sample 
forecast = results.get_forecast(steps=20)

# forecast mean
mean_forcast = forecast.predicted_mean

# Get confidence intervals of forecasts 
confidence_intervals = forecast.conf_int()
'''

###Generating one-step-ahead predictions
# Generate predictions
one_step_forecast = results.get_prediction(start=-30)

# Extract prediction mean
mean_forecast = one_step_forecast.predicted_mean

# Get confidence intervals of  predictions
confidence_intervals = one_step_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate  predictions
print(mean_forecast)


###Plotting one-step-ahead predictions 
# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean predictions
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits,
		 upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()


###Generating Dynamic Forecast 
# Generate predictions
dynamic_forecast = results.get_prediction(start=-30, dynamic=True)

# Extract prediction mean
mean_forecast = dynamic_forecast.predicted_mean

# Get confidence intervals of predictions
confidence_intervals = dynamic_forecast.conf_int()

# Select lower and upper confidence limits
lower_limits = confidence_intervals.loc[:,'lower close']
upper_limits = confidence_intervals.loc[:,'upper close']

# Print best estimate predictions
print(mean_forecast)


###Plotting dynamic forcast

# plot the amazon data
plt.plot(amazon.index, amazon, label='observed')

# plot your mean forecast
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

# shade the area between your confidence limits
plt.fill_between(lower_limits.index, lower_limits, upper_limits, color='pink')

# set labels, legends and show plot
plt.xlabel('Date')
plt.ylabel('Amazon Stock Price - Close USD')
plt.legend()
plt.show()


'''
Population Growth data is non-stationary. To make it stationary we take the difference of 
the time series to make it stationary. We can model it with the ARMA model, and now it is
trained to predict the value of the difference of the time series. What we really want to 
predict is not the difference but the actual value of the time series. We can acheive 
this by carefully transforming our prediction of the differences.

We start with predictions of the difference values. The opposite of taking the difference
is taking the cumulative sum or integral. We will need to use this transform to go from 
predictions of the difference values to prediction of the absolute values. We can do this 
using the numpy-dot-cumsum function. 

# If we apply he cumsum function we now have a prediction of how much the time series 
changed from its initial value over the forecast period. To get an absolute value we 
need to add the last value of the original time series to this. You can plot this to get 
forecat of a non-stationary time series

#Reconstructing originall time series after differencing 
diff_forecast = results.get_forecast(steps=10).predicted_mean
from numpy import cumsum
mean_forecast= cumsum(diff_forecast) + df.iloc[-1,0]

If we would like to plot our uncertainties as before we will need to carefully transform 
them as well. These steps of starting with non-stationary data; differencing to make it 
to make it stationary and then integrating the forecast are very common in time series 
modeling 

The ARIMA model 

* Take the difference 
* Fit ARMA model 
* Integrate forecast 

Can we avoid doing so much work?
Yes
ARIMA - Autoregressive Integrated Moving Average 

from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(df, order=(p, d, q))

p - number of autoregressive lags
d - order of differencing 
q - number of moving average lags 

ARIMA(p, 0, q) = ARMA(p, q)

Using ARIMA model
model = SARIMAX(df, order=(2,1,1))
#Fit model 
model.fit()
#Make forecast
mean_forecast = results.get_forcast(steps=10).predicted_mean

Here we use the difference to the time series data just one time then 
apply an ARMA(2,1) model. This is acheived by using am ARIMMA(2,1,1)
model. 

You only need to take the difference of your data until it is stationary
and no more. We work this out before we apply a model, using the augmented
Dickey-Fuller test to decide the difference order. By the time we coe to 
apply a model we already know the degree of differencing we should apply

adf = adfuller(df.iloc[:,0])
print('ADF Statistic:', adf[0])
print('p-value', adf[1])

Output:
ADF Statistic: -2.674
p-value: 0.0784

adf = adfuller(df.diff().dropna().iloc[:,0])
print('ADF Statistic:', adf[0])
print('p-value:', adf[1])

Output:
ADF Statistic: -4.978
p-value: 2.44e-05
'''

###Differencing and fitting the ARMA

# Take the first difference of the data
amazon_diff = amazon.diff().dropna()

# Create ARMA(2,2) model
arma = SARIMAX(amazon_diff, order=(2,0,2))

# Fit model
arma_results = arma.fit()

# Print fit summary
print(arma_results.summary())


###Unrolling ARMA forecast 
# Make arma forecast of next 10 differences
arma_diff_forecast = arma_results.get_forecast(steps=10).predicted_mean

# Integrate the difference forecast
arma_int_forecast = np.cumsum(arma_diff_forecast)

# Make absolute value forecast
arma_value_forecast = arma_int_forecast + amazon.iloc[-1,0]

# Print forecast
print(arma_value_forecast)


###Fitting an ARIMA model
# Create ARIMA(2,1,2) model
arima = SARIMAX(amazon, order=(2,1,2))

# Fit ARIMA model
arima_results = arima.fit()

# Make ARIMA forecast of next 10 values
arima_value_forecast = arima_results.get_forecast(steps=10).predicted_mean

# Print forecast
print(arima_value_forecast)



'''
Intro to ACF and PACF

One of the main ways to identify the correct model order is by using the autocorrelation
function, the ACF, and the partial autocorrelation function the PACF

ACF - Autocorrelation Function
PACF - Partial autocorrelation function

What is the ACF?
The autocorrelation function at lag-1 is  the correlation between a time series and the same
time series offset by one step 

* lag-1 autocorrelation -> corr(y_t, y_t-1)

The autocorrelation at lag-2 is the correlation between time series and itself offset by two
steps 

* lag-2 autocorrelation -> corr(y_t, y_t-2)

* ...

* lag-n autocorrelation -> corr(y_t, y_t-n)

We can plot the autocorrelation function to get an overview of the data. The bars shoow and ACF
values at increasing lags. If these values are small and lie inside the blue shaded region, then
they are not statistically significant. 


What is PACF?

The partial autocorrelation is the correlation between a time series and the lagged version of itself 
after we subtract the effect of correlation at smaller lags. So it is the correlation that is associated 
with just that particular lag. The partial autocorrelation function is this series of values and we can 
plot it to get another view of the data. 

By comparing the ACF and PCF for time series we can deduce the model order. If the amplitude of the ACF 
tails off with increasing lag and the PACF cuts off after some lag p, then we have an AR(q) model 

AR(p)
____________________________
ACF     Tails off
PACF    Cuts off after lag p



If the amplitude of the ACF cuts off after some lag q and amplitude of the PACF tails off then we have 
a MA(q) model

MA(q)
_____________________________
ACF Cuts off after lag q
PACF Tails off 



If both the aCF and PACF tail off then we have an ARMA model. In this case we can't deduce the model orders 
of p and q from the plot 

ARMA(p, q)
____________________________
ACF     Tails off
PACF    Tails off 

Making plots of ACF and PACF 
Into each function we pass the time series DataFrame and the maximum number of lags we would like to see. We 
also tell it whether to show the autocorrelation at lag-0. Te ACF and PACF at lag-0 will always have a value 
of one so we'll set this argument to false to simplify the plot 

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#create figure 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))

#Make ACF plot
plot_acf(df, lags=10, zero=False, ax=ax1)

#Make PACF plot 
plot_pacf(df, lags, zero=False, ax=ax2)

plt.show()

The time series must be made stationary before making these plots. If the ACF values are high and tail off 
very slowly this is a sign that the data is non-stationarity, so it needs to be differenced. If the autocorrelation 
at lag-1 is very negative this is a sign that we have taken the difference too many times. 
'''

###AR or MA
# Import
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of df
plot_acf(df, lags=10, zero=False, ax=ax1)

# Plot the PACF of df
plot_pacf(df, lags=10, zero=False, ax=ax2)

plt.show()


###Order of earthquakes 
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

# Plot ACF and PACF
plot_acf(earthquake, lags=15, zero=False, ax=ax1)
plot_pacf(earthquake, lags=15, zero=False, ax=ax2)

# Show plot
plt.show()


# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))

# Plot ACF and PACF
plot_acf(earthquake, lags=10, zero=False, ax=ax1)
plot_pacf(earthquake, lags=10, zero=False, ax=ax2)

# Show plot
plt.show()

# Instantiate model
model = SARIMAX(earthquake, order=(1,0,0))

# Train model
results = model.fit()


'''
AIC and BIC 
ACF and PACF cannot be used to choose the order of a model, when both 
of the orders p and q are non-zero. Instead there are other models the 
AIC and the BIC.

The Akaike information criterion, or AIC, is a metric which tells us how 
good a model is. A model which makes better predictions is given a lower AIC
score. The AIC also penalizes models which have lots of parameters. This means 
if we set the order too high compared to the data, we will get a high AIC value 
This stops us overfitting to the training data.

AIC - Akaike information criterion

* Lower AIC indicates a better model 
* AIC likes to choose simple models with lower order 

The Bayesian information criterion, or BIC is very similar to the AIC. Models which
fit the data better have lower BICs and the BIC penalizes over complex models.
For both of these metrics a lower value suggests a better model. The difference 
between these two metics is how much they penalize model complexity 

The BIC penalizes additional model orders more than AIC and so the BIC will sometimes
suggest a simpler model. The AIC and BIC will often choose the same model, but when 
they don't we will have to make a choice. 

If our goal is to identify good predictive models, we should use AIC. However if our goal 
is to identify a good explanatory model, we should use the BIC.

AIC vs BIC
* BIC favors simpler models than AIC
* AIC is better at choosing predictive models 
* BIC is better at choosing good explanatory model 

After fitting a model in python we can find the AIC and BIC by using the summary of the 
fitted-models-results object. These are on  the right of the table. You can also access 
the AIC an BIC directly by using the dot-aic

# Create model 
model = SARIMAX(df, order=(1,0,1))

# Fit model 
results = model.fit()

# Print fit summary 
print(results.summary())

print('AIC:', results.aic)
print('BIC:', results.bic)

Being able to acces the AIC and BIC directly means we can write loops to fit multiple 
ARIMA models to a dataset, to find the best model order 

# Loop over AR order 
for p in range(3):
    # Loop over MA order 
    for q in range(3):
        # Fit model
        model = SARIMAX(df, order=(p,0,q))
        results = model.fit()
        # print the model order and the AIC/BIC values 
        print(p, q, results.aic, results.bic)
        
        
If we want to test a large number of model orders, we can append the model order and the 
AIC and BIC to a list, and later convert it to a DataFrame. This means that we can sort 
through the AIC/BIC score and not have to search through the orders by eye. 

oreder_aic_bic = []
# Loop over AR order 
for p in range(3):
    # Loop over MA order 
    for q in range(3):
    # Fit model
    model = SARIMAX(df, order=(p,0,q))
    results = model.fit()
    # Add order and scores to list 
    order_aic_bic.append((p,q, results.aic, results.bic))
 
 # Make DataFrame of model order and AIC/BIC scores 
 order_df = pd.DataFrame(order_aic_bic, columns=['p', 'q', 'aic', 'bic'])
 
 Sometimes when searching over model orders you will attempt to fit an order that leads to an
 error 
 
 
 # Fit model 
 model = SARIMAX(df, order=(2,0,1))
 results = model.fit()
 
 ValueError: Non-stationary starting autoregressive parameters found with 'enforce_stationarity'
 set to True. 
 
 This is just a bad model for this data, and when we loop over p and q we would like to skip 
 this one 
 
 We can skip these orders in our loop by using a try and except block in python
 
 # Loop over AR order 
 for p in range(3):
    # Loop over MA order
    for q in range(3):
       try:
        # Fit model 
        model = SARIMAX(df, order=(p, 0,q))
        results = model.fit()
        
        # Print the model order and the AIC/BIC values 
        print(p, q, results.aic, results.bic)
       except:
        # Print AIC and BIC as None when fails
        print(p, q, None, None)
'''

### Searching over model order 
# Create empty list to store search results
order_aic_bic=[]

# Loop over p values from 0-2
for p in range(3):
  # Loop over q values from 0-2
    for q in range(3):
      	# create and fit ARMA(p,q) model
        model = SARIMAX(df, order=(p,0,q))
        results = model.fit()
        
        # Append order and results tuple
        order_aic_bic.append((p, q, results.aic, results.bic))



###Choosing order with AIC and BIC
# Construct DataFrame from order_aic_bic
order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p','q','AIC','BIC'])

# Print order_df in order of increasing AIC
print(order_df.sort_values('AIC'))

# Print order_df in order of increasing BIC
print(order_df.sort_values('BIC'))


###AIC and BIC vs ACF and PACF
# Loop over p values from 0-2
for p in range(3):
    # Loop over q values from 0-2
    for q in range(3):
      
        try:
            # create and fit ARMA(p,q) model
            model = SARIMAX(earthquake, order=(p,0,q))
            results = model.fit()
            
            # Print order and results
            print(p, q, results.aic, results.bic)
            
        except:
            print(p, q, None, None)     



'''
Model diagnostics
Introducion to model diagnostics 
After we have picked a final model or a final few models we should ask how 
good they are. This is a key part to the model building life cycle.

To diagnoze our model we focus on the residuals to the training data. The residuals 
are the difference between our model's one-step- ahead predictions and the real values
of the time series. In statsmodels the residuals over the training period can be 
accessed using the dot-resid attribute of the result object 

# Fit model 
model = SARIMAX(df, order=(p,d,q))
resullts = model.fit()
# Assign residuals to variable 
residuals = results.resid

Mean adsolute error 
How far our the predictions from the real values?

mae = np.mean(np.abs(residuals))

For an ideal model the residuals should be uncorrelated white Gaussian noise centered on 
zero. The rest of our diagnostics will help us to see if this is true 

Plot diagnostics
If the model fits well the residuals will be white Gaussian noise. We can use the
result object's dot-plot-underscore-diagnostics method to generate four common plots 
for evaluating this 

Plot diagnostics 
If the model fits well the residuals will be white Gaussian noiise
# Create the 4 diagnostics plots
results.plot_diagnostics()
plt.show()

One of the four plots shows the one-step-ahead 'Standardized residual's 
If our model is working correctly, there should be no obvious structure in 
the residuals. 

Another of the four plots shows, us the distribution of the residuals. The 
orange line shows us a smooth version of this histogram; and the green line 
shows a normal distribution. If our model is good these two lines should be 
almost the same 

The normal Q-Q plot is another way to show how the distribution of the model 
residuals compares to a normal distribution. If our residuals are normally 
distributed then all the points should lie along the red line, except perhaps 
some values at either end. 

The last plot is the correlogram, which is just an ACF plot of the residuals 
rather than the data. 95% of the correlation for lag greater than zero shuld
not be significant. If there is ssignificant correlation in the residuals, 
it means that there is information in the data that our model hasn't captured 

Some of the plots also have accompanying test statistics in results dot-summary 
tables 

print(results.summary())

Prob(Q) is the p-value associated with the null hypothesis that the residuals
have no correlation structure.  Prob(Q) is the p-value associated with the null 
hypothesis that the residuals have no correlation structure. 

Prob(JB) is the p-value associated with the null hypothesis that the residuals 
normally distributed 

If either p-value is less than 0.05 we reject that hypothesis 

*Prob(Q) - p-value for null hypothesis that residuals are uncorrelated 
*Prob(JB) - p-value for null hypothesis that residuals are normally distributed
'''

###Mean absolute error 
# Fit model
model = SARIMAX(earthquake, order=(1,0,1))
results = model.fit()

# Calculate the mean absolute error from residuals
mae = np.mean(np.abs(results.resid))

# Print mean absolute error
print(mae)

# Make plot of time series for comparison
earthquake.plot()
plt.show()

###Diagnostic Summary Statistics
# Create and fit model
model1 = SARIMAX(df, order=(3,0,1))
results1 = model1.fit()

# Print summary
print(results1.summary())

# Create and fit model
model2 = SARIMAX(df, order=(2,0,0))
results2 = model2.fit()

# Print summary
print(results2.summary())

###Plot diagnostics 
# Create and fit model
model = SARIMAX(df, order=(1,1,1))
results=model.fit()

# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()


'''
Best practices framework for using these tools 
The Box-Jenkins method 
The Box-Jenkins method is a kind of a checklist for you to go from raw data 
for production. The three main steps that stand between you and a production 
ready model are identification, estimation and model diagnostics 
The Box-Jenkins method 
From raw data -> production model 
* identification 
* estimation 
* model diagnostics 

In the identification step w explore and characterize the data to find some form 
of it which is appropriate to ARIMA modeling. We need to know whether the time 
series is stationary and find which transformations, such as differencing or taking 
the log of the data, will make it stationary. Once we found a stationary form we must
identify which orders p and q are the most promising. 


Identification 
* Is the time series stationary?
* What differencing will make it stationary?
* What transforms will make it stationary?
* What values of p and q are most promising? 

Tools to test for stationarity include plotting the time series and using the augmented 
Dickey-Fuller test. Then we can take the difference or apply transformations until we 
find the simplest set of transformations that make the time series stationary. Finally
we use the ACF and PACF to identify promising mode orders. 


Plot the time series 
* df.plot()
Use augmented Dicky-Fuller test 
* adfuller()
Use transforms and/or differencing 
* df.diff(), np.log(), np.sqrt()
Plot AVF/PACF
* plot_acf(), plot_pacf()


The next step is estimation, which involves using numerical methods to estimate the AR
and MA coefficients of the data. Thankfully, this is autoatically done for us when we
call the model's dot fit method. At this stage we might fit many modes and use the AIC
and BIC to narrow down to more promising candidates. 



Estimation 
* Use the data to train the model coefficients
* Done for us using model.fit()
* Choose between models using AIC and BIC
    - results.aic, resullts.bic 


In the model diagnostics step, we evaluate the quanity of the best fitting model. Here 
is where we use our test statistics and diagnostic plots to make sure the residuals 
are well behaved. 

Model diagnostics 
* Are the residualls uncorrelated 
* Are residuals normally distributed 
    - results.plot_diagnostics()
    - resullts.summary()

Decision 

Using the information gathered from statistical tests and plots during the diaagnostic 
step, we need to make a decision. Is the model good enough or do we need to go back 
and rework it. 

Repeat 

If the residuals aren't as they sould be we will go back and rethink our choiices in the 
earlier steps    

* We go through the proces again with more information 
* Find a better model 

If the residualls are okay then we can go ahead and make forcasts 

Production 

* Ready to make forcasts
    - results.get_forecast()

Box-Jenkins 
This should be your general project workflo when developing time series models. 
'''

###Idetification 
# Plot time series
savings.plot()
plt.show()

# Run Dicky-Fuller test
result = adfuller(savings['savings'])

# Print test statistic
print(result[0])

# Print p-value
print(result[1])

###Identification II
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
 
# Plot the ACF of savings on ax1
plot_acf(savings, lags=10, zero=False, ax=ax1)

# Plot the PACF of savings on ax2
plot_pacf(savings, lags=10, zero=False, ax=ax2)

plt.show()

###Estimation
# Loop over p values from 0-3
for p in range(4):
  
  # Loop over q values from 0-3
    for q in range(4):
      try:
        # Create and fit ARMA(p,q) model
        model = SARIMAX(savings, order=(p,0,q), trend='c')
        results = model.fit()
        
        # Print p, q, AIC, BIC
        print(p, q, results.aic, results.bic)
        
      except:
        print(p, q, None, None)
 
 ###Diagnostics 
 # Create and fit model
model = SARIMAX(savings, order=(1,0,2), trend='c')
results = model.fit()

# Create the 4 diagostics plots
results.plot_diagnostics()
plt.show()

# Print summary
print(results.summary())


'''
Seasonal time series 

A seasonal time series has predictable patterns that repeat regularly. Although we call 
this feature seasonality, it can repeat after any length of time. These seasonal cycles
might repeat every year like sales of sunscreen, or every week like number of visitors 
to a park, or every day like number of users on a website at any hour. 

Seasonal data 
* Has predictable and repeated patterns 
* Repeats after any amount of time

Seasonal decomposition 
We can think of this, or any time series, as being made of 3 parts. The trend, the seasonal 
component, and the residual. The full time series is these three parts added together 

time series = trend + seasonal + residual

We can use the stats model seasonal_decompose to separate out any time series into these 
three components 


#Import 
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose data
decomp_results = seasonal_decompose(df['IPG3113N'], period=12) # period parameter is the number of datapoints in each cycle

type(decom_results)

#Plot decompoosed data 
decomp_results.plot()
plot.show()

In order to decompose the data we need to know hw often the cycle repeats. Often 
you can guess this but we can also use the ACF to identify the period. 

To find a period we look for a lag greater than one, which is a peak in the ACF 
plot. Here there is a peak at 12 lags and so this means that the seasonal 
component repeats every 12 time steps. 

Sometimes it is hard to tell by eye whether a trend is seasonal or not. This is where 
the ACF is particularly useful. 

Before performing the ACF we need to make it stationary. We have detrended time series 
before by taking the difference. However, this time we are only trying to find the period
of the time. 

#Subtract long rolling average over N steps 
df = df - df.rolling(N).mean() #N is the window size

#Drop NaN values 
df = df.dropna()

# Create figure 
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

#Plot ACF
plot_acf(df.dropna(), ax=ax, lags=25, zero=False)
plt.show()
'''

###Seasonal Decompose 
# Import seasonal decompose
from statsmodels.tsa.seasonal import seasonal_decompose

# Perform additive decomposition
decomp = seasonal_decompose(milk_production['pounds_per_cow'], 
                            period=12)

# Plot decomposition
decomp.plot()
plt.show()

###Seasonal ACF and PACF
# Create figure and subplot
fig, ax1 = plt.subplots()

# Plot the ACF on ax1
plot_acf(water['water_consumers'], lags=25 , zero=False,  ax=ax1)

# Show figure
plt.show()

###

# Subtract the rolling mean
water_2 = water - water.rolling(15).mean()

# Drop the NaN values
water_2 = water_2.dropna()

# Create figure and subplots
fig, ax1 = plt.subplots()

# Plot the ACF
plot_acf(water_2['water_consumers'], lags=25, zero=False, ax=ax1)

# Show figure
plt.show()

'''
SARIMA models 
A SARIMA or seasonal ARIMA model is the tool of choice for seasonal time series 
Previously we saw that we could split up our time series into a seasonal and 
some on-seasonal components. Fitting a SARIMA model is like fitting two different 
ARIMA models at once, one to the seasonal part an another to the non-seasonal part. 

Since we have these two models we will have two sets of orders. We have 
non-seasonal orders for the autoregressive, difference and moving average parts. We 
also have this set of orders for the seasonal part   

We use P, D, and Q for these seasonal orders. There is also a new order, S, which is 
the length of the seasonal cycle

The SARIMA model 
Seasonal ARIMA = SARIMA                         SARIMA(p,d,q)(P,D,Q)S
* Non-seasonal orders                           *Seasonal Orders 
    - p:autoregressive order                        -P: seasonal autoregressive order 
    - d: differencing order                         -D: seasonal differencing order 
    - q: moving average order                       -Q:seasonal moving average order 
    
The SARIMA model 
ARIMA(2,0,1) model:
                        y_t = a_1 * y_t-1 + a_2 * y_t-2 + m_1 * epselon_t-1 + epselon_t
                        
This is the equation for a simple ARIMA model. We regress the time series model against 
itself at lags-1 and 2 and against the shock at lag-1

SARIMA(0,0,0)(2,0,1)_7 model:
                        y_t = a_7 * y_t-7 + a_14 * y_t-14 + m_7 * epselon_t-7 + epselon_t

This is the equaion for a simple SARIMA model with season length of 7 days. This SARIMA 
model only has a seasonal part; we have set the non-seasonal orders to zero. We regress the 
ime series against itself at lags one season and two seasons and against the shock at lag 
of one season. This particular SARIMA model will be able to capture seasonal, weekly patters,
but won't be able to capture local, day to day patterns. 

If we construct a SARIMA mdel and include non-seasonal orders as well, then we can capture 
both of these patterns 

Fitting a SARIMA model 

# Imports 
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Instatiate model
model = SARIMAX(df, order=(p,d,q), seasonal_order=(P,D,Q,S))

# Fit model
results = model.fit()

Seaonal Differencing 
Subtract the time series value of one season ago 
    delta(y_t) = y_t - y_t-S
    
# Take the seasonal difference 
df_diff = df.diff(S) #length of the seasonal cycle 

If the time series hows a trend then we take the normal difference. If there is a strong
seasonal cycle, then we will also take the seasonal difference. Once we have found 
the two orders of differencing, and made the time series stationary we need to find
the other model orders. 

To find the non-seasonal orders, we plot the ACF and PACF of the differenced time series

To find the seasonal orders we plot the ACF an PACF of the differenced time series at multiple 
seasonal steps. Then we can use the same table of ACF and PACF rulels to work out the seasonal 
order.    

Plotting seasonal ACF and PACF 
# Create figure 
fig, (ax1, ax2) = plt.subplots(2,1)

# Plot seasonal ACF
plot_acf(df_diff, lags=[12, 24, 36, 48, 60, 72], ax=ax1)

# Plot seasonal PACF
plot_pacf(df_diff, lags=[12, 24, 36, 48, 60, 72], ax=ax2) 

plt.show()

'''

###Fitting SARIMA models 
# Create a SARIMAX model
model = SARIMAX(df1, order=(1,0,0), seasonal_order=(1,1,0,7))

# Fit the model
results = model.fit()

# Print the results summary
print(results.summary())

###

# Create a SARIMAX model
model = SARIMAX(df2, order=(2,1,1), seasonal_order=(1,0,0,4))

# Fit the model
results = model.fit()

# Print the results summary
print(results.summary())

###

# Create a SARIMAX model
model = SARIMAX(df3, order=(1,1,0), seasonal_order=(0,1,1,12))

# Fit the model
results = model.fit()

# Print the results summary
print(results.summary())

###

# Create the figure 
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))

# Plot the ACF on ax1
plot_acf(aus_employment_diff, lags=11, zero=False, ax=ax1)

# Plot the PACF on ax2
plot_pacf(aus_employment_diff, lags=11, zero=False, ax=ax2)

plt.show()

###

# Make list of lags
lags = [12, 24, 36, 48, 60]

# Create the figure 
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(8,6))

# Plot the ACF on ax1
plot_acf(aus_employment_diff, lags=lags, zero=False, ax=ax1)

# Plot the PACF on ax2
plot_pacf(aus_employment_diff, lags=lags, zero=False, ax=ax2)

plt.show()

###SARIMA vs ARIMA forecast
# Create ARIMA mean forecast
arima_pred = arima_results.get_forecast(steps=25)
arima_mean = arima_pred.predicted_mean

# Create SARIMA mean forecast
sarima_pred = sarima_results.get_forecast(steps=25)
sarima_mean = sarima_pred.predicted_mean

# Plot mean ARIMA and SARIMA predictions and observed
plt.plot(dates, sarima_mean, label='SARIMA')
plt.plot(dates, arima_mean, label='ARIMA')
plt.plot(wisconsin_test, label='observed')
plt.legend()
plt.show()


'''
Automating and saving 
Previouslly we searched over ARIMA model order using for-loops. Now that we
we have seasonal orders as well, this is very complex. Fortunately there is
a package that will do most of this work for us. This is the pmdarima package 
The auto_arima function from this package looks over model orders to find
the best one. The object returned by this function is the results object of 
the best model fouond by the search. This object is almost exactly like a 
statsmodels SARIMAX

import pmdarima as pmdarima

results = pm.auto_arima(df)

print(results.summary())

results.plot_diagnostics()

Non-seasonal search parameters 

results = pm.auto_arima(df,         # only required argument is data 
                        d=0,        # non-seasonal difference order
                        start_p=1,   # initial guss fpr p
                        start_q=1,  # initial guess for q
                        max_p=3,    # max value of p to test 
                        max_q=3,    # max value of q to test 
                        seasonal=True,   # is the time series seasonal
                        m=7,        # the seasonal period
                        D=1,        # seasonal difference order    
                        start_P=1,  # initial guess for P
                        start_Q=1,  # initial guess for Q
                        max_P=2,    # max value of P to test
                        max_Q=2,    # max value of Q to test
                        information_criterion='aic',    # used to select best model
                        trace=True  # print results whilst training 
                        error_action='ignore',      # ignore orders that don't work
                        stepwise=True)              # apply intelligent order search  


Once you have fit a model in this way, you may want to save it and load it later 
To save the model we use the dump function from the joblib package. We pass the 
model results object an the filepath into this function.

# Import 
import joblib

# Select a filepath 
filepath = 'localpath/great_model.pkl'

# Save model to filepath 
joblib.dump(model_results_object, filepath)

# Select a filepath 
filepath = 'localpath/great_model.pkl'

# Load model object from filepath 
model_results_object = joblib.load(filepath)


In the case that time has passed since we trained the saved model, we may want to 
incorporate data that we have collected since then. This isn't the same as choosing 
the model order again and so if you are updating with a large amount of new data it 
may be the best method to go back to the start of the Box-Jenkins method. Updating 
time seires models with new data is really important since tthey use the most recent 
available data for future predictions. 


# Add new observations and update parameters
model_results_object.update(df_new)
'''

###Automated model selection

# Create auto_arima model
model1 = pm.auto_arima(df1,
                      seasonal=True, m=7,
                      d=0, D=1, 
                 	  max_p=2, max_q=2,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model1.summary())


##

# Create model
model2 = pm.auto_arima(df2,
                      d=1,
                      seasonal=False,
                      trend='c',
                 	  max_p=2, max_q=2,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model2.summary())

##

# Create model for SARIMAX(p,1,q)(P,1,Q)7
model3 = pm.auto_arima(df3,
                      seasonal=True, m=7,
                      d=1, D=1, 
                      start_p=1, start_q=1,
                      max_p=1, max_q=1,
                      max_P=1, max_Q=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True) 

# Print model summary
print(model3.summary())


###Saving and Updating Models 
# Import joblib
import joblib

# Set model name
filename = "candy_model.pkl"

# Pickle it
joblib.dump(model, filename)

##

# Import
import joblib

# Set model name
filename = "candy_model.pkl"

# Load the model back in
loaded_model = joblib.load(filename)

##

# Update the model
loaded_model.update(df_new)

'''
SARIMA and Box-Jenkins

We previously covered the Box-Jenkins method for ARIMA models 
We go through identification of the mode order; estimating or fitting the 
model; diagnosing the model residuals; and finally production. For SARIMA 
models the only step in the method which will change is the identification 
step 

ARIMA time series               
Identification                  
Estimation                      
Model Diagnostics 
Model Okay?
Yes
Production


At the identification step we add the tasks of determining whether a time 
series is seasonal, and if so, then finding its seasonal period. We also need 
to consider transforms to make seasonal time series stationary, such as
seasonal and non-seasonal differencing and other transforms. 

Box-Jenkins with seasonal data 
* Determine if time series is seasonal 
* Find seasonal period 
* Find transforms to make data stationary 
    - Seasonal and non-seasonal differencing 
    - Other transforms 

Sometimes we will have the choice of whether to apply seasonal differencing,
non-seasonal differencing or both to make a time series stationary. Some good 
rules of thumb are that you should never use more than one order of seasonal 
differencing and never more than two orders of differencing in total. 

Mixed differencing 
* D should be 0 or 1
* d + D should be 0-2


When you have a stron seasonal pattern, you should always use one order of seasonal 
differencing. This will ensure that the seasonal oscillation will remain in your 
dynamic predictions far into the future withut fading out. Just like in ARIMA 
modeling sometimes we need to use other transformations on our time series before
fitting. Whenever the seasonality is additive we shouldn't need to apply any 
transforms except differencing. Additive seasonality is wwhere the seasonal 
pattern just adds ortakes away a little from the trend. When the seasonality is 
multiplicative, the SARIMA model can't fit this without extra transforms. If 
the seasonality is multiplicative the smplitude of seasonal oscillations will
get larger as the data trends up or smaller as it trends down. To deal with 
this we take the log transform of the data before modeling it. 

When the seasonality is multiplicative, the SARIMA model can't fit this without
extra transforms.  

* Additive series = trend + season 
* Proceed as usual with differencing 

* multiplicative series = trend x season
* Apply log transform first - np.log 
'''

###SARIMA model diagnostics 
# Import model class
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Create model object
model = SARIMAX(co2, 
                order=(1,1,1), 
                seasonal_order=(0,1,1,12), 
                trend='c')
# Fit model
results = model.fit()

###
# Plot common diagnostics
results.plot_diagnostics()
plt.show()


###SARIMA forcast
# Create forecast object
forecast_object = results.get_forecast(steps=136)

# Extract predicted mean attribute
mean = forecast_object.predicted_mean

# Calculate the confidence intervals
conf_int = forecast_object.conf_int()

# Extract the forecast dates
dates = mean.index

plt.figure()

# Plot past CO2 levels
plt.plot(co2.index, co2, label='past')

# Plot the prediction means as line
plt.plot(mean.index, mean, label='predicted')

# Shade between the confidence intervals
plt.fill_between(mean.index, conf_int.iloc[:,0], conf_int.iloc[:,1], alpha=0.2)

# Plot legend and show figure
plt.legend()
plt.show()

# Print last predicted mean
print(mean.iloc[-1])

# Print last confidence interval
print(conf_int.iloc[-1])