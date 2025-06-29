###Correlation and AutoCorrelation
'''
Introduction to Time Series Analysis Using Python
Qunatopia is place to discuss code 

Goals of the Course
* Learn about time series models 
* Fit data to a time series model 
* Use the models to make forcasts of the future 
* Learn how to use the relevent statistical packages in Python 

to_datetime() is used to convert an index, often read in as a string, 
into a datetime index 

Changing an index to datetime 
    df.index = pd.to_datetime(df.index)

Plotting data
    df.plot()
    
Daily data can be converted to weekly data with resample method
ges and differences of a time series 


*Computing percent chan
df['col'].pct_change()
df['col'].diff()

Pandas correlation method of Series 
df['ABC'].corr(df['XYZ'])
'''

###A "Thin" Application of Time Series 

# Import pandas and plotting modules
import pandas as pd
import matplotlib.pyplot as plt

# Convert the date index to datetime
diet.index = pd.to_datetime(diet.index)

# From previous step
diet.index = pd.to_datetime(diet.index)

# Plot the entire time series diet and show gridlines
diet.plot(grid=True)
plt.show()

# From previous step
diet.index = pd.to_datetime(diet.index)

# Slice the dataset to keep only 2012
diet2012 = diet['2012']

# Plot 2012 data
diet2012.plot(grid=True)
plt.show()

###Merging Time Series With Different Dates 
# Import pandas
import pandas as pd

# Convert the stock index and bond index into sets
set_stock_dates = set(stocks.index)
set_bond_dates = set(bonds.index)

# Take the difference between the sets and print
print(set_stock_dates - set_bond_dates)

# Merge stocks and bonds DataFrames using join()
stocks_and_bonds = stocks.join(bonds, how='inner')

'''
Often two time series vary together 
A correlation coefficient is a measure of how much two series vary together 
A correlation of 1 means that the two series have a perfect linear relationship
with no deviations.
High correlations mean that the two series vary strongly together 
Low correlation means they vary together, but there is a weak association 
A Negative correlation means they vary in opposite directions, but still with a 
linear relationship

Compute the correlation between two financial series 

First step: Compute percentage chages of both series 
df['SPX_Ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_prices'].pct_change()

Visualize the correlation with a scatter plot 
plt.scatter(df['SPX_Ret'], df['R2000_Ret'])
plt.show()

Use pandas correlation method for the Series 

correlation = df['SPX_Ret'].corr(df['R2000_Ret'])
print("Correlation is: ", correlation)
'''

###Correlation of Stocks and Bonds 
# Compute percent change using pct_change()
returns = stocks_and_bonds.pct_change()

# Compute correlation using corr()
correlation = returns['SP500'].corr(returns['US10Y'])
print("Correlation of stocks and interest rates: ", correlation)

# Make scatter plot
plt.scatter(returns['SP500'],returns['US10Y'])
plt.show()

###Flying Suacers Aren't Correlated to Flying Markets 
# Compute correlation of levels
correlation1 = levels['DJI'].corr(levels['UFO'])
print("Correlation of levels: ", correlation1)

# Compute correlation of percent changes
changes = levels.pct_change()
correlation2 = changes['DJI'].corr(changes['UFO'])
print("Correlation of changes: ", correlation2)

'''
Simple linear regression y_t = alpha +beta(x_t) + epselon_t

Linear Regression is also known as Ordinary Least Squares (OLS)

Python packages to perform regressions 
*In statsmodels:
import statsmodels.api as sm 
sm.OLS(y, x).fit()

*In numpy:
np.polyfit(x, y, deg=1)

*In pandas:
pd.ols(y, x)

*In scipy:
from scipy import stats 
stats.linregress(x, y)

Beware: the order of x and y is not consistent across packages

Example: Regression of Small Cap Returns on Large Cap 

*Import the statsmodels module 
import statsmodels.api as sm 

*As before, compute percentage changes in both series 
df['SPX_Ret'] = df['SPX_Prices'].pct_change()
df['R2000_Ret'] = df['R2000_Prices'].pct_change()

*Add a constant to the DataFrame for the regression intercept 
df = sm.add_constant(df)

#By adding a column of 1s, the tats models will compute the 
regression coefficient of that column as well, which can be 
interpreted as the intercept of the line 

#First row returns NaN

*Delete the row of NaN
df = df.dropna()

*Run the regression 
results = sm.OLS(df['R2000_Ret'], df[['const', 'SPX_Ret']]).fit()
print(results.summary())

Regression output
intercept in results.params[0]
slope in results.params[1] 
Another statistic to take note of is the R-Squared of 0.753

From the scatter diagrams, you saw that the correlation measures how closely the 
data are clustered along a line.
The R-squared also measures how well the linear regression fits the data.
So as youo would expect, there is a relationship between correlation and 
R-squared  

*[corr(x,y)]^2 = R^2 

The sign of the correlation is the sign of the slope of the regression line
If the regression line is positively sloped, the correlation is positive 
If the regression line is negatively sloped, the correlation is negative 

0.753 is a positive therfore the slope is positvie 

The correlation = sqrt(0.753) = 0.868
'''

###Looking at a Regression's R-Squared 
# Import the statsmodels module
import statsmodels.api as sm

# Compute correlation of x and y
correlation = x.corr(y)
print("The correlation between x and y is %4.2f" %(correlation))

# Convert the Series x to a DataFrame and name the column x
dfx = pd.DataFrame(x, columns=['x'])

# Add a constant to the DataFrame dfx
dfx1 = sm.add_constant(dfx)

# Regress y on dfx1
result = sm.OLS(y, dfx1).fit()

# Print out the results and look at the relationship between R-squared and the correlation above
print(result.summary())

'''
Autocorrelation is the correlation of a single time series with a lagged copy of itself. It's also called "serial correlation".
Often when we refer to a serie's correlation, we mean the "lag-one" autocorrelation, also called serial corial correlation.

What do positive and negative correlations mean? 
With financial time series, when returns have a negative autocorrelation, we say it is "mean reverting"
  -Mean Reversion - Negative autocorrelation

Alternatively, if a series has a poositive autocorrelation, we say it is "trend-following". 
  -Momentum or Trend Following - Positive autocorrelation 
  
Traders Use Autocorrelation to Make Money 
* Individual stocks 
    - Historically have negative autocorrelation 
    - Measured oer short horizons (days)

Example of Positive Autocorrelation: Exchange Rates 
- Use daily yen/dollar exchange rates in DataFrame df from FRED (Federal Reserve Economic Data)
- Convert Index to datetime 

# Convert index to datetime 
df.index = pd.to_datetime(df.index)
# Downsample from daily to monthly data
df.resample(rule='M, how='last')
# Compute returns from prices 
df['Return'] = df['Price'].pct_change()
# Compute autocorrelation 
autocorrelation = df['return'].autocorr()
print("The autocorrelation is: ", autocorrelation) 
'''

###A popular strategy using autocorrelation 
# Convert the daily data to weekly data
MSFT = MSFT.resample(rule='W').last()

# Compute the percentage change of prices
returns = MSFT.pct_change()

# Compute and print the autocorrelation of returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly returns is %4.2f" %(autocorrelation))

###Are Interest Rates Autocorrelated? 
# Compute the daily change in interest rates 
daily_diff = daily_rates.diff()

# Compute and print the autocorrelation of daily changes
autocorrelation_daily = daily_diff['US10Y'].autocorr()
print("The autocorrelation of daily interest rate changes is %4.2f" %(autocorrelation_daily))

# Convert the daily data to annual data
yearly_rates = daily_rates.resample(rule='A').last()

# Repeat above for annual data
yearly_diff = yearly_rates.diff()
autocorrelation_yearly = yearly_diff['US10Y'].autocorr()
print("The autocorrelation of annual interest rate changes is %4.2f" %(autocorrelation_yearly))


'''
Autocorrelation Function 
The sample autocorrelation function, or ACF, shows not only the lag one correlation, but the entire 
autocorrelation function for different lags.
Any significant non-zero autocorrelations implies that the series can be forecast from the past 
Autocorrelation Function 
* Autocorrelation Function (ACF): The autocorrelaton as a function of the lag.
An ACF can also be useful for selecting parsimonious model for fitting the data. 

In this example, the pattern of the autocorrelation suggests a model for the series that will be 
discussed in the next chapter. 

plot_acf is the statsmodel function for plotting the autocorrelation function 
* Import module:
    from statessmodels.graphics.tsaplots import plot_acf
* Plot the ACF: 
    plot_acf(x, lags= 20, alpha=0.05)
* The alpha argument sets the width of the confidence interval, for example: if alpha equal 0.05, that 
means that if the true autocorrelation at that lag is zero, there is only a 5% chance the sample 
autocorrelation will fall outside that window. You will get a wider confidence interval if you set alpha 
lower, or if you have fewer observations. An approximation to the width of the 95% confidence intervals, 
if you make some simplifying assumptions, is plus or minus 2 over the square root of the number of
observations in your series. If you don't want to see confidence intervals in your plot set alpha equal 
to one. 

Confidence Interval of ACF
* Argument alpha sets the width of the confidence interval 
* Example: alpha=0.05
    - 5% chance that if true autocorrelation is zero, it will fall outside blue band
* confidence bonds are wider if: 
    - lower alpha 
    - fewer observations
* Under some simplifying assumptions, 95% confidence bands are +/- 2 / sqrt(N)
* If you want no bands on plot, set alpha=1


Besides plotting the ACF, you can also extract its numerical values usng a similar Python function, acf,
instead of plot acf.

from statsmodels.tsa.stattools import acf
print(acf(x))

Even if the true autocorrelations were zero at all lags, in a finite sample of returns you won't see the 
estimate of the autocorrelations exactly zero. In fact, the standard deviation of the sample 
autocorrelation is 1/sqrt(N) where N is the number of observations, so if N = 100, for example, 
the standard deviation of the ACF is 0.1, and since 95% of a normal curve is between +1.96 and -1.96 
standard deviations from the mean, the 95% confidence interval is +/-1.96/sqrt(N). This approximation only holds when 
the true autocorrelations are all zero.
'''

###Taxing Exercie: Compute the ACF
# Import the acf module and the plot_acf module from statsmodels
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf

# Compute the acf array of HRB
acf_array = acf(HRB)
print(acf_array)

# Plot the acf function
plot_acf(HRB, alpha=1)
plt.show()

###Are we Confident This Stock Mean is Reverting?
# Import the plot_acf module from statsmodels and sqrt from math
from statsmodels.graphics.tsaplots import plot_acf
from math import sqrt

# Compute and print the autocorrelation of MSFT weekly returns
autocorrelation = returns['Adj Close'].autocorr()
print("The autocorrelation of weekly MSFT returns is %4.2f" %(autocorrelation))

# Find the number of observations by taking the length of the returns DataFrame
nobs = len(returns)

# Compute the approximate confidence interval
conf = 1.96/sqrt(nobs)
print("The approximate confidence interval is +/- %4.2f" %(conf))

# Plot the autocorrelation function with 95% confidence intervals and 20 lags using plot_acf
plot_acf(returns, alpha=0.05, lags=20)
plt.show()


'''
Although people define white noise slightly differently, a general definition is that 
it is a series with mean that is constant with time, a variance that is also constant with time,
and zero autocorrelation at all lags. There are several special cases of white Noise. 
Fo example, if the data is white noiise but also has a normal, 

What is White Noise? or Gaussian, distribution, then it is called Gaussian White Noise
* White Noise is a series with:
    - Constant mean 
    - Constant variance 
    - Zero autocorrelations at all lags 
    - Special Case: If data has normal distribution, then Gaussian White Noise

numpy random normal creates an array of normally distributed random numbers. The loc argument is 
the mean and the scale argument is the standard deviation. This is one way to generate a white 
noise series. And all of the autocorrelations of a white noise series are zero. The returns on 
the stock market are pretty close to a white noiise process 

Simulated White Noise 
* It's very easy to generate white noise 
    import numpy as np
    noise = np.random.normal(loc=0, scale=1, size=500)
    plt.plot(noise)
* Autocorrelation of White Noise 
plot_acf(noise, lags=50)
'''

###Can't forcast noise 
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Simulate white noise returns
returns = np.random.normal(loc=0.02, scale=0.05, size=1000) #mean = loc, stdev = scale

# Print out the mean and standard deviation of returns
mean = np.mean(returns)
std = np.std(returns)
print("The mean is %5.3f and the standard deviation is %5.3f" %(mean,std))

# Plot returns series
plt.plot(returns)
plt.show()

# Plot autocorrelation function of white noise returns
plot_acf(returns, lags=20)
plt.show()


'''
Random Walk
What is a Random Walk?
* Today's Price = Yesterday's Price + Noise
    P_t = P_t-1 + epselon_t
    
* The change in price of a random walk is just white noise
    P_t - P_t-1 = epselon_t

* If prices are in logs, then the difference in log prices is one way to measure returns 

The bottom line is that if stock *prices* follow a random walk, then stock *returns* are 
White Noise 

* Can't forcast a random walk
* Best guess for tomorrow's price is simply today's price
Today's Price = Yesterday's Price + Noise 
    P_t = P_t-1 + epselon_t

in a random walkk with drift, prices on average drift by mu every period 
* Random walk with drift:
    P_t = mu + P_t-1 + epselon_t

The change is pice for a random walk with drift is still white noise but with a mean of mu 
* Change in price is white noiise with non-zero mean: 
    P_t - P_t-1 = mu + epselon_t
    
So if we now think of stock prices as a random walkk with drift, then the returns are still 
white noise, but with an average return of mu instead of zero. 

Statistical Test for Random Walk

To test whether a series like stock prices follows a random walk, you can regress current 
prices on agged prices, if the slope coefficient, beta, is not significantly different from one
then we CANNOT reject the null hyptothesis that the series is a random walk. However, 
if the slope coefficient is significantly less than one, then we CAN reject the null hypothesis 
that the series is a random walk

* Random walk with drift 
    P_t = mu + P_t-1 + epselon_t
*Regression test for random walk 
    P_t = alpha + beta(P_t-1) + epselon_t
* Test:

An identical way to do that test is to regress the difference in prices on the lagged price, 
and instead of testing whether the slope coeffieciet is 1, now we test whether it is zero.
This is called the "Dickey-Fuller" test. If you add more lagged prices on the right had side,
then it's called the Augmented Dickey-Fuller test. 

* Regression test for random walk 
    P_t - P_t-1 = alpha + beta(P_t-1) + epselon_t
* Test H_0: Beta = 0(random walk)


Statsmodels has a function, adfuller, for performing the Augmented Dickey-Fuller test 
ADF Test in Pyton 
* import module from statsmodels 
    from statsmodels.tsa.stattools import adfuller 
* Run Augmented Dickey-Test 
    adfuller(x)
    
Example: Is the S&P500 a Random Walk?
# Run Augmented Dickey-Fulller Test on SPX data
results = adfuler(df['SPX'])

# Print p-values
print(results[1])

The main output we're interested in is the p-value of the test

If the p-value is less than 5%, we can reject the null hypothesis that the series is 
a random walk with 95% confidence 

In this case the p-value is much higher than 0.05, it is 0.78

Therefore, we cannot reject the null hypothesis that the S&P500 is a random walk. You can 
also print out the full output of the test, which gives other information, like the number
of observations (1257), the test statistic (-0.917) and the critical values of the test 
statistic for various alphas - 1%, 10% and 5%
'''

###Generate a Random Walk 
# Generate 500 random steps with mean=0 and standard deviation=1
steps = np.random.normal(loc=0, scale=1, size=500)

# Set first element to 0 so that the first price will be the starting stock price
steps[0]=0

# Simulate stock prices, P with a starting price of 100
P = 100 + np.cumsum(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk")
plt.show()

###Get the drift 
# Generate 500 random steps
steps = np.random.normal(loc=0.001, scale=0.01, size=500) + 1

# Set first element to 1
steps[0]=1

# Simulate the stock price, P, by taking the cumulative product
P = 100 * np.cumprod(steps)

# Plot the simulated stock prices
plt.plot(P)
plt.title("Simulated Random Walk with Drift")
plt.show()


###Are Stock Prices a Random Walk?
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Run the ADF test on the price series and print out the results
results = adfuller(AMZN['Adj Close'])
print(results)

# Just print out the p-value
print('The p-value of the test on prices is: ' + str(results[1]))

###How About Stock Returns?
# Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Create a DataFrame of AMZN returns
AMZN_ret = AMZN.pct_change()

# Eliminate the NaN in the first row of returns
AMZN_ret = AMZN_ret.dropna()

# Run the ADF test on the return series and print out the p-value
results = adfuller(AMZN_ret['Adj Close'])
print('The p-value of the test on returns is: ' + str(results[1]))


'''
What is Stationarity?
* Strong stationarity: entire distribution of data is time-invariant 
In joint distribution of the observations do not depend on time

*Weak stationarity: mean, variance and autocorrelation are time-invariant 
(i.e., for autocorrelation, corr(X_t, X_t-tau) is only a function of Tau)
The autocorrelation, the correlation between X-t and X(t-tau) is only a 
function of the lag tau, not a function of time 

Modeling involves estimating a set of parameters, and if a process is not 
stationary and the parameters are different at each point in time, then 
there are too many parameters to estimate, you may end up having more 
parameters than actual data. So stationarity is necessary for a 
parsimonious model, one with a smaller set of parameters to estimate. 
A random walk is a common type of non-stationary series. The variance 
grows with time. For example, if stock prices are a random walk, then 
the uncertainty of our prices tomorrow is much less than the uncertainty 
10 years from now. Seasonal series are also non-stationary.           
 
Why do we care?
* If parameters vary with time, too many parameters to estimate 
* Can only estimate a parsimonious model with a few parameters 

A random walk is non stationary 
plt.plot(SPY)

If you take the difference, the new series is White Noise, which is stationary
plt.plot(SPY.diff()) 

For series that are growing exponentially as well as exhibiting a strong 
seasonal pattern. 
First, if you take only the log of the series, you eliminate the 
exponentiial growth. 
But if you take both the log of the series and then the seasonal difference, 
in the lower right, the transformed series looks stationary.      
'''

###Seasonal Adjustment During Tax Season 
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Seasonally adjust quarterly earnings
HRBsa = HRB.diff(4)

# Print the first 10 rows of the seasonally adjusted series
print(HRBsa.head(10))

# Drop the NaN data in the first four rows
HRBsa = HRBsa.dropna()

# Plot the autocorrelation function of the seasonally adjusted series
plot_acf(HRBsa)
plt.show()

'''
Mathematical Description of AR(1) Model 
R_t = mu + phi(R_t-1) + epselon_t

Since there is only one lagged value on the right hand side, 
this is called an AR model of order 1, or simply an AR(1) model. Or simply 
An AR(1) model  
*AR parameter is phi
    If phi is 1 then the process is a random walk 
    If phi is 0, then the process is white noise

In order for the process to be stable and stationary, phi has to be between 
-1 and +1.

Suppose R_t is a time series of stock returns. If phi is negative, 
then a positive return last period, at time t-1, implies that this 
period's return is more likely to be negative. This was referred 
to as "mean reversion" in Chapter 1. On the other hand isf phi is 
positive, then a positive return last period implies that this 
period's return is expected to be positive. This was referred to as 
momentum 

R_t = mu + R_t-1 + epselon_t
Negative phi: Mean Reversion 
Positive phi: Momentum 

Hgher Order AR Models 
* AR(1)
    R_t = mu + phi(R_t-1) + epselon_t
* AR(2)
    R_t = mu + phi(R_t-1) + phi(R_t-2) + epselon_t
* AR(3)
    R_t = mu + phi(R_t-1) + phi(R_t-2) + phi(R_t-3) + epselon_t



The convention is a little counterintuitive: You must include the zero-lag
coefficient of 1, and the sign of the other coefficient is the opposite
of what we have been using. For example for an AR(1) process with phi 
equal to plus 0.9, the second element of the ar array should be the 
opposite sign, -0,9. This is consistant with the time series literature 
in the field of signal processing.
 
from statsmodels.tsa.arima_process import ArmaProcess 
ar = np.array([1, -0.9])
ma = np.array([1])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=1000)
plt.plot(simulated_data)
'''

###Simulate AR(1) Time Series 
# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: AR parameter = +0.9
plt.subplot(2,1,1)
ar1 = np.array([1, -0.9])
ma1 = np.array([1])
AR_object1 = ArmaProcess(ar1,ma1)
simulated_data_1 = AR_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: AR parameter = -0.9
plt.subplot(2,1,2)
ar2 = np.array([1, 0.9])
ma2 = np.array([1])
AR_object2 = ArmaProcess(ar2,ma2)
simulated_data_2 = AR_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)
plt.show()

###Compare the ACF for Several AR Time Series 
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot 1: AR parameter = +0.9
plot_acf(simulated_data_1, alpha=1, lags=20)
plt.show()

# Plot 2: AR parameter = -0.9
plot_acf(simulated_data_2, alpha=1, lags=20)
plt.show()

# Plot 3: AR parameter = +0.3
plot_acf(simulated_data_3, alpha=1, lags=20)
plt.show()

'''
Stats models has another module for estimating the parameters of a given 
AR model. import ARMA, which is a class, and create an instance of that 
class called mod with the arguments being the data that you're trying to 
fit, and the order of the model. The order (1,0) means you're fitting the 
data to an AR(1) model. An order (2, 0) would mean you're fitting the data
to an AR(2) model. The second part of the order is the MA part, which will 
be discussed in the next chapter. Once you instantiate the class, you can 
use the method fit estimate the model, and store the result in result.

To see the full output, use the summary method on result. 

from statsmodels.tsa.arima_model import ARMA
mod = ARMA(simulated_data, order=(1,0))
result = mod.fit()

*Full output (true mu = 0 and phi = 0.9)
    print(result.summary())
    
print(results.params)

If you just want to see the coefficients rather than the entire regression 
output, you can use the params property which returns an array of the filled
coefficients, mu and phi in this case. To do forecasting, both in sample 
and out of sample, you still create an instance of the class ARMA, and you 
use the fit method just as you did to *estimate* the 
parameters. 


* To estimate parameters from data (simulated)
    from statsmodels.tsa.arima_model import ARMA
    mod = ARMA(simulated_data, order=(1,0))
    results = mod.fit()
    
from statsmodels.tsa.arima_model import ARMA
mod = ARMA(simulated_data, order=(1,0))
res = mod.fit()
res.plot_predict(start='2016-07-01, end='2017-06-01')
plt.show()

Now, use the method plot_predict to do forecasting. You give it the start and
end data points for fitting. If the index of the data is a DatetimeIndex object as 
it is here, you can pick dates for the start and end date. 

from statsmodels.tsa.arima_model import ARMA
mod = ARMA(simullated_data, order=(1,0))
res = mod.fit()
res.plot_predict(start='2016-07-01', end='2017-06-01')
plt.show()
'''

###Estimating an AR Model 
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Fit an AR(1) model to the first simulated data
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()

# Print out summary information on the fit
print(res.summary())

# Print out the estimate for the constant and for phi
print("When the true phi=0.9, the estimate of phi (and the constant) are:")
print(res.params)

###Forcasting with an AR Model 
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast the first AR(1) model
mod = ARMA(simulated_data_1, order=(1,0))
res = mod.fit()
res.plot_predict(start=10, end=1000)
plt.show()

###Let's Forcast Interest Rates 
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast interest rates using an AR(1) model
mod = ARMA(interest_rate_data, order=(0,1))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start=0, end='2022')
plt.legend(fontsize=8)
plt.show()

###Compare AR Model with Random Walk 
# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot the interest rate series and the simulated random walk series side-by-side
fig, axes = plt.subplots(2,1)

# Plot the autocorrelation of the interest rate series in the top plot
fig = plot_acf(interest_rate_data, alpha=1, lags=12, ax=axes[0])

# Plot the autocorrelation of the simulated random walk series in the bottom plot
fig = plot_acf(simulated_data, alpha=1, lags=12, ax=axes[1])

# Label axes
axes[0].set_title("Interest Rate Data")
axes[1].set_title("Simulated Random Walk Data")
plt.show()

'''
Choosing the Right Model 

Identifying the Order of an AR Model 
*The order of an AR(p) model will usually be unknown 
*Two techniques to determine order 
    -PArtial Autocorrelation Function 
    -Information criteria 
    
Partial Autocorrelation Function (PACF)
R_t = phi(0,1) + phi(1,1)(R_t-1) + epselon_lt
R_t = phi(0,2) + phi(1,2)(R_t-1)+phi(2,2)(R_t-2) + epselon_2t
R_t = phi(0,2) + phi(1,2)(R_t-1)+phi(2,2)(R_t-2) + phi(3,3)(R_t-3) + epselon_3t
R_t = phi(0,2) + phi(1,2)(R_t-1)+phi(2,2)(R_t-2) + phi(3,3)(R_t-3) + phi(4,4)(R_t-4) + epselon_4t

Each new ph(N,1) represents how significant adding a lag when there are 2, 3 or 4 lags

Plot PACF in Python
    * Same as ACF, but use plot_pacf instead of plt_acf
    * Import module
        from statsmodels.graphics.tsaplots import plot_pacf
    * Plot the PACF
        plot_pacf(x, lags=20, alpha=0.05)

The more parameters in a model, the better the model will fit the data. But this can lead to overfitting 
of the data. 

Information Criteria 
    *Information criteria: adjusts goodness-of-fit for number of parameters
    *Two popular adjusted goodness-of-fit measures 
        -AIC (Akaine Information Criterion)
        -BIC (Bayesian Information Criterion)
        
        
Getting Information Criteria From 'statesmodels'

* You learned earlier how to fit an AR model 
    from statsmodels.tsa.arima_model import ARMA
    mod = ARA(simulated_data, order=(1,0))
    result = mod.fit()
* And to get full output 
    result.summary()
* Or just the parameters    
    result.params
* To get the AIC and BIC
    result.aic
    result.bic
Fit several models
Choose the one with the lowest Baysian information criterion

Example 
Fit a simulated AR(3) to different AR(p) models
Choose p with the lowest BIC
'''

###Estimate Order of Model PACF
# Import the modules for simulating data and for plotting the PACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_pacf

# Simulate AR(1) with phi=+0.6
ma = np.array([1])
ar = np.array([1, -0.6])
AR_object = ArmaProcess(ar, ma)
simulated_data_1 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(1)
plot_pacf(simulated_data_1, lags=20)
plt.show()

# Simulate AR(2) with phi1=+0.6, phi2=+0.3
ma = np.array([1])
ar = np.array([1, -0.6, -0.3])
AR_object = ArmaProcess(ar, ma)
simulated_data_2 = AR_object.generate_sample(nsample=5000)

# Plot PACF for AR(2)
plot_pacf(simulated_data_2, lags=20)
plt.show()

###Estimate Order of Model: Information Criteria
# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
for p in range(7):
    mod = ARMA(simulated_data_2, order=(p,0))
    res = mod.fit()
# Save BIC for AR(p)    
    BIC[p] = res.bic
    
# Plot the BIC as a function of p
plt.plot(range(1,7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Bayesian Information Criterion')
plt.show()



'''
Describe Model 

Mathematical Description of MA(1) Model 
R_t = mu + epselon_t + theta_e_t-1

*Since only one lagged error on the right hand side this is called: 
-MA model of order 1, or 
-MA(1) models 
MA models are stationary for all values of data 


If theta is negative then poositive shock last period, represented by epsilon
t-1, would have caused last period's return to be positive, but
return is more likely to be negative. 

Negative theta: one-period mean reversion

A shock two periods ago would have no effect on today's return - only the shock now
and last period

Positive theta: one-period momentum

Also note that the lag-1 autocorrelation turns out to be theta, but theta over 1 plus
theta squared 

Not: one-period autocorrelation is theta/(1 + theta^2), not theta

So far we have seen MA(1) models 

MA(1)
    R_t = mu + epselon_t - theta_1 epselon_t-1
MA(2)
    R_t = mu + epselon_t - theta_1 epselon_t-1 - theta_2 epselon_t-2 
MA(3)
    R_t = mu + epselon_t - theta_1 epselon_t-1 - theta_2 epselon_t-2 - theta_3 epselon_t-3
...

The model can be extended to include more lagged errors and moretheta parameters

Just like in the last chapter with AR models, you may want to simulate a pure MA 
process 

You can use the same statsmodels module, ArmaProcess

from statsmodels.tsa.arima_process import ArmaProcess
ar = np.array([1])
ma = np.array([1, 0.5])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=1000)
plt.plot(simulated_data)

This time, for MA(1), the AR order is just an array containing 1 and the MA(1) parameter
theta. Unlike with AR simulation, you don't need reverse the sign of theta

To simulate data, use the method generate_sample, with the number of simulated
samples as an argument 
'''

###Simulate MA(1) Time Series 
# import the module for simulating data
from statsmodels.tsa.arima_process import ArmaProcess

# Plot 1: MA parameter = -0.9
plt.subplot(2,1,1)
ar1 = np.array([1])
ma1 = np.array([1, -0.9])
MA_object1 = ArmaProcess(ar1, ma1)
simulated_data_1 = MA_object1.generate_sample(nsample=1000)
plt.plot(simulated_data_1)

# Plot 2: MA parameter = +0.9
plt.subplot(2,1,2)
ar2 = np.array([1])
ma2 = np.array([1, 0.9])
MA_object2 = ArmaPro2 etc . cess(ar2, ma2)
simulated_data_2 = MA_object2.generate_sample(nsample=1000)
plt.plot(simulated_data_2)

plt.show()

###Compute the ACF for Several MA Time Series 

'''
Unlike an AR(1), an MA(1) model has no autocorrelation beyond lag 1, an MA(2) model
has no autocorrelation begond lag 1, an MA(2) model ha no autocorrelation beyond lag
1, an MA(2) model has no autocorrelation beyond lag 2 etc.
'''

# Import the plot_acf module from statsmodels
from statsmodels.graphics.tsaplots import plot_acf

# Plot 1: MA parameter = -0.9
plot_acf(simulated_data_1, lags=20)
plt.show()

# Plot 2: MA parameter = 0.9
plot_acf(simulated_data_2, lags=20)
plt.show()

# Plot 3: MA parameter = -0.3
plot_acf(simulataed_data_3, lags=20)
plt.show()


'''
Estimating and Forecasting an MA Model 
Estimating an MA Model 
* Same as estimating an AR model (except order=(0,1))
    from statsmodels.tsa.arima_model import ARMA
    mod = ARMA(simulated_data, order=(0,1))
    result = mod.fit()
    
The same module that you used to estimate an AR model can be used to estimate the 
parameters of an MA model. 

Now the order is (0,1), for an MA(1), not (1,0) for an AR(1)


Forcasting an MA Model 
from stasmodel.tsa.arima_model import ARMA
mod = ARMA(simullated_data, order=(0,1))
res = mod.fit()
res.plot_predict(start='2016-07-01', end='2017-06-01')
plt.show()

One thing to note with an MA model unlike and AR model, all forcast beyond the 
one-step ahead forecast will be the same.
'''

###Estimating the MA model
# Import the ARMA module from statsmodels
from statsmodels.tsa.arima_model import ARMA

# Forecast the first MA(1) model
mod = ARMA(simulated_data_1, order=(0,1))
res = mod.fit()
res.plot_predict(start=990, end=1010)
plt.show()


'''
ARMA models 
Here is the formula for an ARMA(1,1) model, which has the familiar AR(1) and 
MA(1) components

R_t = mu + phi R_t-1 + epselon_t + theta_t-1

ARMA models can be converted to pure AR or pure MA models

*Converting AR(1) into an MA(infinity)

Here is an example of converting an AR(1) model into a MA(infinity) model

R_t = mu + phi R_t-1 + epselon_t # AR(1) model

R_t = mu + phi(mu + phi R_t-2 + epselon_t-1) + epselon_t #AR(1) equation is substituted for R_t-1
.
.
.
R_t = (mu/1-phi) + epselon_t + phi epselon_t-1 - phi^2 epselon_t-2 + phi^3 epselon_t-3 + ...
'''

###High Frequecy Stock Prices
# import datetime module
import datetime

# Change the first date to zero
intraday.iloc[0,0] = 0

# Change the column headers to 'DATE' and 'CLOSE'
intraday.columns = ['DATE', 'CLOSE']

# Examine the data types for each column
print(intraday.dtypes)

# Convert DATE column to numeric
intraday['DATE'] = pd.to_numeric(intraday['DATE'])

# Make the `DATE` column the new index
intraday = intraday.set_index('DATE')


###More Data Cleaning: Missing Data
# Notice that some rows are missing
print("If there were no missing rows, there would be 391 rows of minute data")
print("The actual length of the DataFrame is:", len(intraday))

# Everything
set_everything = set(range(391))

# The intraday index as a set
set_intraday = set(intraday.index) 

# Calculate the difference
set_missing = set_everything - set_intraday

# Print the difference
print("Missing rows: ", set_missing)

# Fill in the missing rows
intraday = intraday.reindex(range(391), method='ffill')

# Change the index to the intraday times
intraday.index = pd.date_range(start='2017-09-01 9:30', end='2017-09-01 16:00', freq='1min')

# Plot the intraday time series
intraday.plot(grid=True)
plt.show()


###Applying an MA Model
# Import plot_acf and ARMA modules from statsmodels
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARMA

# Compute returns from prices and drop the NaN
returns = intraday.pct_change()
returns = returns.dropna()

# Plot ACF of returns with lags up to 60 minutes
plot_acf(returns, lags=60)
plt.show()

# Fit the data to an MA(1) model
mod = ARMA(returns, order=(0,1))
res = mod.fit()
print(res.params)

###Equivalence of AR(1) and MA(infinity)

# import the modules for simulating data and plotting the ACF
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf

# Build a list MA parameters
ma = [0.8**i for i in range(30)]

# Simulate the MA(30) model
ar = np.array([1])
AR_object = ArmaProcess(ar, ma)
simulated_data = AR_object.generate_sample(nsample=5000)

# Plot the ACF
plot_acf(simulated_data, lags=30)
plt.show()

'''
Cointegration Models
* Two series, P_t and Q_t can be random walks
* But the linear combination P_t - c Q_t may not be a random walk!
If thats true
- P_t - c Q_t is forcastable
- P_t and Q_t are said to be cointegrated 

Even in the prices of two different assets still follow random walks, it is still 
both follow random walks, it is still possible that a linear combination of them 
is not a random walk. If that is true even though P and Q are not that forcastable 
becasue they are random walks, the linear combination is forcastable, and we say
that P and Q are cointegrated.

Analogy: Dog on a leash 
P_t = Owner 
Q_t = Dog 

- Both series look like a random walk
- Difference, or distance between them, looks mean reverting
    If dog falls too far behind, it gets pulled forward 
    If dog gets too far ahead, it gets pulled back 
    
You can break down the process for testing whether two series are cointegrated into
two steps. First, you regress the level of one series on the level of the other series, 
to get the slope coefficient c. Then you run the Augmented Dickey-Fuller test, 
the test for a random walk (linear combination of two series). 

Alternatively, statsmodels has a function coint that combines both steps 

Two steps for cointegration 
* Regress P_t on Q_t and get slope c
* Run augmented Dickey-Fuller test on P_t - c Q_t to test for random walk
* Alternatively, can use coint function in statsmodels that combines both steps 

from statsmodels.tsa.stattools import coint
coint(P, Q)
'''

###A dog on a leash
# Plot the prices separately
plt.subplot(2,1,1)
plt.plot(7.25*HO, label='Heating Oil')
plt.plot(NG, label='Natural Gas')
plt.legend(loc='best', fontsize='small')

# Plot the spread
plt.subplot(2,1,2)
plt.plot(7.25*HO-NG, label='Spread')
plt.legend(loc='best', fontsize='small')
plt.axhline(y=0, linestyle='--', color='k')
plt.show()

###A dog on a leash (Part 2)
 # Import the adfuller module from statsmodels
from statsmodels.tsa.stattools import adfuller

# Compute the ADF for HO and NG
result_HO = adfuller(HO['Close'])
print("The p-value for the ADF test on HO is ", result_HO[1])
result_NG = adfuller(NG['Close'])
print("The p-value for the ADF test on NG is ", result_NG[1])

# Compute the ADF of the spread
result_spread = adfuller(7.25 * HO['Close'] - NG['Close'])
print("The p-value for the ADF test on the spread is ", result_spread[1])

###Are bitcoint and ethereum cointegrated 
# Import the statsmodels module for regression and the adfuller function
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Regress BTC on ETH
ETH = sm.add_constant(ETH)
result = sm.OLS(BTC,ETH).fit()

# Compute ADF
b = result.params[1]
adf_stats = adfuller(BTC['Price'] - b*ETH['Price'])
print("The p-value for the ADF test is ", adf_stats[1])


'''
Analyzing Temperature Data 
* Temperature data:
    - New York City from 1870-2016
    - Downloaded from National Oceanic and Atmospheric Administration (NOAA)
    - Convert index to datetime object 
    - Plot data
    - Test for Random Walk 
    - Take first differences # transforms it into stationary series 
    - Compute Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    - Fit a few AR, MA, and ARMA models 
    - Use Information Criterion to choose best model 
    - Forecast temperature over next 30 years 
'''

###Is Temperature a Random Walk (with Drift)?
# Import the adfuller function from the statsmodels module
from statsmodels.tsa.stattools import adfuller

# Convert the index to a datetime object
temp_NY.index = pd.to_datetime(temp_NY.index, format='%Y')

# Plot average temperatures
temp_NY.plot()
plt.show()

# Compute and print ADF p-value
result = adfuller(temp_NY['TAVG'])
print("The p-value for the ADF test is ", result[1])


###Getting "Warmed" Up: Look at Autocorrelations
# Import the modules for plotting the sample ACF and PACF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Take first difference of the temperature Series
chg_temp = temp_NY.diff()
chg_temp = chg_temp.dropna()

# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)

# Plot the ACF
plot_acf(chg_temp, lags=20, ax=axes[0])

# Plot the PACF
plot_pacf(chg_temp, lags=20, ax=axes[1])
plt.show()

###Which ARMA Model is Best?
# Import the module for estimating an ARMA model
from statsmodels.tsa.arima_model import ARMA

# Fit the data to an AR(1) model and print AIC:
mod_ar1 = ARMA(chg_temp, order=(1, 0))
res_ar1 = mod_ar1.fit()
print("The AIC for an AR(1) is: ", res_ar1.aic)

# Fit the data to an AR(2) model and print AIC:
mod_ar2 = ARMA(chg_temp, order=(2, 0))
res_ar2 = mod_ar2.fit()
print("The AIC for an AR(2) is: ", res_ar2.aic)

# Fit the data to an ARMA(1,1) model and print AIC:
mod_arma11 = ARMA(chg_temp, order=(1, 1))
res_arma11 = mod_arma11.fit()
print("The AIC for an ARMA(1,1) is: ", res_arma11.aic)

###Don't throw ou that winter coat yet 
# Import the ARIMA module from statsmodels
from statsmodels.tsa.arima_model import ARIMA

# Forecast temperatures using an ARIMA(1,1,1) model
mod = ARIMA(temp_NY, order=(1,1,1))
res = mod.fit()

# Plot the original series and the forecasted series
res.plot_predict(start='1872-01-01', end='2046-01-01')
plt.show()


'''
Advanced Topics 
* GARCH Models
* Nonlinear Models
* Multivariate Time Series Models 
* Regime Switching Models 
* State Space Models and Kalman filtering 
'''