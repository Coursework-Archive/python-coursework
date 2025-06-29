import pandas as pd
#returns DataFrame
type(AAPL)
#returns (num of rows, num of colms)
AAPL.shape

#returns the column names
AAPL.columns

#returns the data type of AAPL.columns
pandas.indexes.base.Index

#returns the type of index
type(AAPL.index)

#slice dataArrays
AAPL.iloc[:5,:]
#slices the data frame from the slice to the 5th row using .iloc

AAPL.iloc[-5:,:]
#slices the data from the 5 last row to the end of the data using .iloc

#see just the top rows to the data
AAPL.head(5)

#returns other useful summary informations
AAPL.info()

#Assigning scalar value to column slice broadcasts value to each row
import numpy as np
AAPL.iloc[::3, -1] = np.nan
#assigns a scalar value 'nan' or not a number. The slice consists of every
#3 row starting from 0 in the last column

AAPL.head(6)

low = AAPL['low']
type(low)
#returns a series

#returns a numpy array is one dimensional
low.values
type(low)
numpy.ndarray

#DataFrame is a two dimensional array that has columns that are series

# Import numpy
import numpy as np

# Create array of DataFrame values: np_vals
np_vals = df.values

# Create new array of base 10 logarithm values: np_vals_log10
np_vals_log10 = np.log10(np_vals)

# Create array of new DataFrame by passing df to np.log10(): df_log10
df_log10 = np.log10(df)

# Print original and new data containers
[print(x, 'has type', type(eval(x))) for x in ['np_vals', 'np_vals_log10', 'df', 'df_log10']]


##Building DataFrames from scratch
import pandas as pd
users = pd.read_csv('datasets/users.csv', index_col=0)
print(users)

#building from lists
data = dict(zipped)
users = pd.DataFrame(data)
print(users)

#proadcasting a technique used in numpy and pandas
users['fees'] = 0 # Broadcasts to entire column
print(users)

#Broadcasting with a dictionary
import pandas as pd
heights = [ 59.0, 65.2, 62.9, 65.4, 63.7, 65.7, 64.1]
data = {'height': heights, 'sex': 'M'}
results = pd.DataFrame(data)
print(results)


# Zip the 2 lists together into one list of (key,value) tuples: zipped
zipped = list(zip(list_keys, list_values))

# Inspect the list using print()
print(zipped)

# Build a dictionary with the zipped list: data
data = dict(zipped)

# Build and inspect a DataFrame from the dictionary: df
df = pd.DataFrame(data)
print(df)

# Build a list of labels: list_labels
list_labels = ['year', 'artist', 'song', 'chart weeks']

# Assign the list of labels to the columns attribute: df.columns
df.columns = list_labels

# Make a string with the value 'PA': state
state = 'PA'

# Construct a dictionary: data
data = {'state':state, 'city':cities}

# Construct a DataFrame from dictionary data: df
df = pd.DataFrame(data)

# Print the DataFrame
print(df)

#Reading DataFrames from files

import pandas as pd
filepath = 'ISSN_D_tot.csv'
#header=None tells python to not interpret the first row as header labels, gives the columns numbers
sunspots = pd.read_csv(filepath, header=None)
sunspots.info()

#access a slice of the middle of the DataFrame
sunspots.iloc[10:20, :]

#you can provide a list of column names ad pass it into the .read_csv method
#you can change the values of the column that are non-values by using the na_values assignment
#You can combine date columns using al ist of list with the pars_dates argument to identify the columns to combine
col_names = ['year', 'month', 'day', 'dec_date', 'sunspots', 'definite']
sunspots = pd.read_csv(filepath, header=None, names=col_names, na_values={'sunspots':[' -1']}, parse_dates=[[0, 1, 2]])
sunspots.iloc[10:20, :]

#you can import the data to a new csv/excel file
out_csv = 'sunspots.csv'
sunspots.to_csv(out_csv)

out_tsv = 'sunspots.tsv'
sunspots.to_csv(out_tsv, sep='\t')

out_xlsx = 'sunspots.xlsx'
sunspots.to_excel(out_xlsx)

# Read in the file: df1
df1 = pd.read_csv(data_file)

# Create a list of the new column labels: new_labels
new_labels = ['year', 'population']

# Read in the file, specifying the header and names parameters: df2
df2 = pd.read_csv(data_file, header=0, names=new_labels)

# Print both the DataFrames
print(df1)
print(df2)

# Read the raw file as-is: df1
df1 = pd.read_csv(file_messy)

# Print the output of df1.head()
print(df1.head(5))

# Read in the file with the correct parameters: df2
df2 = pd.read_csv(file_messy, delimiter=' ', header=3, comment='#')

# Print the output of df2.head()
print(df2.head())

# Save the cleaned up DataFrame to a CSV file without the index
df2.to_csv(file_clean, index=False)

# Save the cleaned up DataFrame to an excel file without the index
df2.to_excel('file_clean.xlsx', index=False)


#Data visualization

import pandas as pd
import matplotlib.pyplot as plt
aapl = pd.read_csv('aapl.csv', index_col='date', parse_dates=True)

aapl.hea(6)

close_arr = aapl['close'].values
type(close_arr)

plt.plot(close_arr)

plt.show()

#plotting pandas Series directly
close_series = aapl['close']
type(close_series)
plt.plot(close_series)
plt.show()

#Better method for plotting
close_series.plot()
plt.show()

#DaataFrame plotting method
aapl.plot()
plt.show()

#plots all columns at once
plt.plot(aapl)
plt.show()

#fixing scales
aapl.plot()
plt.yscale('log') #logrithmc scale on vertical axis
plt.show()

#customizing plots
aapl['open'].plot(color='b', style='.-', legend=True)
aapl['close'].plot(color='r', style='.', legend=True)
plt.axis(('2001', '2002', 0, 10))
plt.show()

aapl.loc['2001':'2004', ['open', 'close', 'high', 'low']].plot()
plt.savefig('aapl.png')
plt.savefig('aapl.jpg')
plt.savefig('aapl.pdf')
plt.show()


# Create a plot with color='red'
df.plot(color='red')

# Add a title
plt.title('Temperature in Austin')

# Specify the x-axis label
plt.xlabel('Hours since midnight August 1, 2010')

# Specify the y-axis label
plt.ylabel('Temperature (degrees F)')

# Display the plot
plt.show()

# Plot all columns (default)
df.plot()
plt.show()

# Plot all columns as subplots
df.plot(subplots=True)
plt.show()

# Plot just the Dew Point data
column_list1 = ['Dew Point (deg F)']
df[column_list1].plot()
plt.show()

# Plot the Dew Point and Temperature data, but not the Pressure data
column_list2 = ['Temperature (deg F)','Dew Point (deg F)']
df[column_list2].plot()
plt.show()

#Visual exploratory data analysis
import pandas as pd
import matplotlib.pyplot as plt
iris = pd.read_csv('iris.csv', index_col=0)
print(iris.shape)

#load the DataFrame and explore with .head
iris.head()
iris.plot(x='sepal_length', y='sepal_width')
plt.show()

#with for dimensional data you can enter graph it more meaning fully in this way
iris.plot(x='sepal_length', y='sepal_width', kind='scatter')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

#individual variable distribution are more informative, whiske box plot
iris.plot(y='sepal_length', kind='box')
plt.ylabel('sepal width (cm)')
plt.show()

#histogram, this is an example of a probability distribution function
#bins (integer): number of intervals or bins
#range (tuple): extrema of bins (minimum, maximum)
#normed (boolean): whether to normalize to one
#cumulative (boolean): compute Cumulative Distribution Function (CDF)
iris.plot(y='sepal_legth', kind='hist')
plt.xlabel('sepal legth (cm)')
plt.show()


iris.plt(y='sepal_legth', kind='hist', bins=30, range=(4,8), normed=True)
plt.xlabel('sepal length (cm)')
plt.show()

#Cumulative distribution
iris.plot(y='sepal_legth', kind='hist', bins=30, range=(4,8), cumulative=True, normed=True)
plt.xlabel('sepal length (cm)')
plt.title('Cumulative distribution function (CDF)')
plt.show()

#three different plot idioms for histograms
iris.plt(kind='hist')
iris.plt.hist()
iris.hist()

# Create a list of y-axis column names: y_columns
y_columns = ['AAPL', 'IBM']

# Generate a line plot
df.plot(x='Month', y=y_columns)

# Add the title
plt.title('Monthly stock prices')

# Add the y-axis label
plt.ylabel('Price ($US)')

# Display the plot
plt.show()

# Generate a scatter plot
df.plot(kind='scatter', x='hp', y='mpg', s=sizes)

# Add the title
plt.title('Fuel efficiency vs Horse-power')

# Add the x-axis label
plt.xlabel('Horse-power')

# Add the y-axis label
plt.ylabel('Fuel efficiency (mpg)')

# Display the plot
plt.show()

# Make a list of the column names to be plotted: cols
cols = ['weight','mpg']

# Generate the box plots
df[cols].plot(kind='box', subplots=True)

# Display the plot
plt.show()

# This formats the plots such that they appear on separate rows
fig, axes = plt.subplots(nrows=2, ncols=1)

# Plot the PDF
df.fraction.plot(ax=axes[0], kind='hist', bins=30, normed=True, range=(0,.3))
plt.show()

# Plot the CDF
df.fraction.plot(bins=30, ax=axes[1], kind='hist', cumulative=True, normed=True, range=(0,.3))
plt.show()

#Statistical exploratory data analysis
iris.describe() # summary statistics

#using describe to analyze your data
#counts include non-null values
iris['sepal_legth'].count() #Applied to Series

#Series method count returns a scalar integer
iris['sepal_width'].count() # Applied to Series

iris[['petal_legth', 'petal_width']].count() # Applied to DataFrame

type(iris[['petal_legth', 'petal_width']].count()) # returns Series

iris['sepal_legth'].mean() # Applied to Series

iris.mean() # Applied to entire DataFram, column wise ignoring null entries

iris.std()
#bell curve mean at the middle, standard deviation is the width

iris.median()

q = 0.5
iris.quantile(q) #quantile are percentiles, the median is the 50th percentile


#inter-quartile range (IQR)
q = [0.25, 0.75]
iris.quantile(q)

#ranges
iris.min()
iris.max()

# Print the minimum value of the Engineering column
print(df['Engineering'].min())

# Print the maximum value of the Engineering column
print(df['Engineering'].max())

# Construct the mean percentage per year: mean
mean = df.mean(axis='columns')

# Plot the average percentage per year
mean.plot()

# Display the plot
plt.show()

# Print summary statistics of the fare column with .describe()
print(df['fare'].describe())

# Generate a box plot of the fare column
df.fare.plot(kind='box')

# Show the plot
plt.show()

# Print the number of countries reported in 2015
print(df['2015'].count())

# Print the 5th and 95th percentiles
print(df.quantile([0.05, 0.95]))

# Generate a box plot
years = ['1800','1850','1900','1950','2000']
df[years].plot(kind='box')
plt.show()

# Print the mean of the January and March data
print(january.mean(), march.mean())

# Print the standard deviation of the January and March data
print(january.std(), march.std())

#filtering by species
indices = iris['species'] == 'setosa'
setosa = iris.loc[indices,:] # extract new DataFrame
indices = iris['species'] == 'versicolor'
versicolor = iris.loc[indices,:] # extract new DataFrame
indices = iris['species'] == 'virginica'
virginica = iris.loc[indices,:] # extract new DataFrame

#separating populations
#describe species column
iris['species'].describe()
    #count # non-null entries
iris['species'].unique()    #unique # distinct values
    #top most frequent category
    #freq # occurences of top
#check indexes
    setosa.head(2)
#computing errors
    error_setosa = 100 * np.abs(describe_setosa - describe_all)
    error_setosa = error_setosa/describe_setosa

# Compute the global mean and global standard deviation: global_mean, global_std
global_mean = df.mean()
global_std = df.std()

# Filter the US population from the origin column: us
us = df[df['origin'] == 'US']

# Compute the US mean and US standard deviation: us_mean, us_std
us_mean = us.mean()
us_std = us.std()

# Print the differences
print(us_mean - global_mean)
print(us_std - global_std)

# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=3, ncols=1)

# Generate a box plot of the fare prices for the First passenger class
titanic.loc[titanic['pclass'] == 1].plot(ax=axes[0], y='fare', kind='box')

# Generate a box plot of the fare prices for the Second passenger class
titanic.loc[titanic['pclass'] == 2].plot(ax=axes[1], y='fare', kind='box')

# Generate a box plot of the fare prices for the Third passenger class
titanic.loc[titanic['pclass'] == 3].plot(ax=axes[2], y='fare', kind='box')

# Display the plot
plt.show()

#read_csv() function
##can read strings into datetime objects
##need to specify 'parse_dates=True'
##ISO 8601 format
##yyyy-mm-dd hh:mm:ss

import pandas as pd
sales = pd.read_csv('sales-feb-2015.csv', parse_dates=True, index_col= 'Date')
sales.head() #view the date column
sales.info() #DatetimeIndex list the range of dates

#.loc accessor can be used to select data by row, column
sales.loc['2015-02-19 11:00:00', 'Company']
#here there is a row that is specified and no columns, all of the columns will be returned
sales.loc['2015-2-5']

#alternative formats to select Febuary 5th 2015
sales.loc['February 5, 2015']
sales.loc['2015-Feb-5'']
sales.loc['2015-2'] #whole month
sales.loc['2015'] #whole year

#Can parse between partial data strings using a colon separation
sales.loc['2015-2-16':'2015-2-20']
#can select all of the rows within a 4 day window
evening_2_11 = pd.to_datetime(['2015-2-11 20:00','2015-2-11 21:00', '2015-2-11 22:00', '2015-2-11 23:00'])
evening_2_11

#reindexing providing a new index and matching data as required
sales.reindex(evening_2_11)

#can override the deault behavior for filling nan
#ffill for forward fill
sales.reindex(evening_2_11, method='ffill')
#bfill for ackward fill
sales.reindex(evening_2_11, method='bfill')

# Prepare a format string: time_format
time_format = '%Y-%m-%d %H:%M'

# Convert date_list into a datetime object: my_datetimes
my_datetimes = pd.to_datetime(date_list, format=time_format)

# Construct a pandas Series using temperature_list and my_datetimes: time_series
time_series = pd.Series(temperature_list, index=my_datetimes)

# Extract the hour from 9pm to 10pm on '2010-10-11': ts1
ts1 = ts0.loc['2010-10-11 21:00:00':'2010-10-11 22:00:00']

# Extract '2010-07-04' from ts0: ts2
ts2 = ts0.loc['2010-07-04']

# Extract data from '2010-12-15' to '2010-12-31': ts3
ts3 = ts0.loc['2010-12-15':'2010-12-31']

# Reindex without fill method: ts3
ts3 = ts2.reindex(ts1.index)

# Reindex with fill method, using forward fill: ts4
ts4 = ts2.reindex(ts1.index, method="ffill")

# Combine ts1 + ts2: sum12
sum12 = ts1 + ts2

# Combine ts1 + ts3: sum13
sum13 = ts1 + ts3

# Combine ts1 + ts4: sum14
sum14 = ts1 + ts4

#Resampling time series data
#extracting time series properties with other properties
import pandas as pd
sales = pd.read-csv('sales-feb-2015.csv', parse_dates=True, index_col='Date')
sales.head()

#resampling
#Statistical methods over different time intervals
#mean(), sum(), count(), etc.
#Downsampling
#Reduce datetime rows to slower frequency
#Upsampling
#increasing datetime rows to faster frequency
daily_mean = sales.resample('D').mean() #D stands for daily
daily_mean

#check the daily mean result for February 2nd
print(daily_mean.loc['2015-2-2'])

print(sales.loc['2015-2-2','Units'])

sales.loc['2015-2-2', 'Units'].mean()

sales.resample('D').sum()

#method chaining
sales.resample('D').sum().max()

sales.resample('W').count() #W stands for week
#min, T - minute
#H - hour
#D - day
#B - business day
#W - week
#M - month
#Q - quarter
#A - year

sales.loc[:,'Units'].resample('2W').sum()

#Upsampling from daily to hourly
two_days = sales.loc['2015-2-4': '2015-2-5', 'Units']
two_days

two_days.resample('4H').ffill() #sample every four hours
#fills in the number of sales using the forward fill method
#The technical term is interpolation

# Downsample to 6 hour data and aggregate by mean: df1
df1 = df['Temperature'].resample('6H').mean()

# Downsample to daily data and count the number of data points: df2
df2 = df['Temperature'].resample('D').count()

# Extract temperature data for August: august
august = df['Temperature'].loc['2010-08']

# Downsample to obtain only the daily highest temperatures in August: august_highs
august_highs = august.resample('D').max()

# Extract temperature data for February: february
february = df['Temperature'].loc['2010-02']

# Downsample to obtain the daily lowest temperatures in February: february_lows
february_lows = february.resample('D').min()

#when using the rolling method, you must first use the roling method and then chain after it

# Extract data from 2010-Aug-01 to 2010-Aug-15: unsmoothed
unsmoothed = df['Temperature']['2010-08-01':'2010-08-15']

# Apply a rolling mean with a 24 hour window: smoothed
smoothed = unsmoothed.rolling(window=24).mean()

# Create a new DataFrame with columns smoothed and unsmoothed: august
august = pd.DataFrame({'smoothed':smoothed, 'unsmoothed':unsmoothed})

# Plot both smoothed and unsmoothed data using august.plot().
august.plot()
plt.show()

# Extract the August 2010 data: august
august = df['Temperature']['2010-08']

# Resample to daily data, aggregating by max: daily_highs
daily_highs = august.resample('D').max()

# Use a rolling 7-day window with method chaining to smooth the daily high temperatures in August
daily_highs_smoothed = daily_highs.rolling(window=7).mean()
print(daily_highs_smoothed)

import pandas as pd
sales = pd.read_csv('sales-feb-2015.csv', parse_dates=['Date'])
sales.head()

sales['Company'].str.upper()

#str method can be used to identify which rows contain the substring
#find all of the rows that are hardware or software
sales['Product'].str.contains('ware')

sales['Product'].str.contains('ware').sum()

sales['Date'].dt.hour

#convert between time zones
central = sales['Date'].dt.tz_localize('US/Central')
central

population = pd.read_csv('world_population.csv', parse_dates=True, index_col= 'Date')
population

#Extracts the first value between every decade
population.resample('A').first()

population.resample('A').first().interpolate('linear')

# Strip extra whitespace from the column names: df.columns
df.columns = df.columns.str.strip()

# Extract data for which the destination airport is Dallas: dallas
dallas = df['Destination Airport'].str.contains('DAL')

# Compute the total number of Dallas departures each day: daily_departures
daily_departures = dallas.resample('D').sum()

# Generate the summary statistics for daily Dallas departures: stats
stats = daily_departures.describe()

# Reset the index of ts2 to ts1, and then use linear interpolation to fill in the NaNs: ts2_interp
ts2_interp = ts2.reindex(ts1.index).interpolate(how='linear')

# Compute the absolute difference of ts1 and ts2_interp: differences
differences = np.abs(ts1 - ts2_interp)

# Generate and print summary statistics of the differences
print(differences.describe())

# Build a Boolean mask to filter for the 'LAX' departure flights: mask
mask = df['Destination Airport'] == 'LAX'

# Use the mask to subset the data: la
la = df[mask]

# Combine two columns of data to create a datetime series: times_tz_none
times_tz_none = pd.to_datetime( la['Date (MM/DD/YYYY)'] + ' ' + la['Wheels-off Time'] )

# Localize the time to US/Central: times_tz_central
times_tz_central = times_tz_none.dt.tz_localize('US/Central')

# Convert the datetimes from US/Central to US/Pacific
times_tz_pacific = times_tz_central.dt.tz_convert('US/Pacific')

print(times_tz_central)
print(times_tz_pacific)

#Time series visualization
#Line types
#plot types
#Subplots

import pandas as pd
import matplotlib.pyplot as plt
sp500 = pd.read_csv('sp500.csv', parse_dates=True, index_col='Date')
sp500.head()

#pandas plot
sp500['Close'].plot()
plt.show()
plt.ylabel('Closing Price (US Dollars)')
plt.show()

#one week accessor
sp500.loc['2012-4-1':'2012-4-7', 'Close'].plot(title='S&P500')
plt.ylabel('Closing Price (US Dollars)')
plt.show()
#by default pandas uses a blue line

#to modify the default behavior
sp500.loc['2012-4', 'Close'].plot(style='k.-', title='S&P500')
plt.ylabel('Closing Price (US Dollars)')
plt.show()

#style format string
#color (k:black)
#marker (.:dot)
#line type (-:solid)
#color   marker     line
#b:blue  o:circle    :dotted
#g:green *:star      -:dashed
#r:red   s:square
#c:cyan  +:plus

sp500['close'].plot(kind='area', title='S&P 500')
plt.ylabel('Closing Price (US Dollars)')
plt.show()

sp500.loc['2012', ['Close','Volume']].plot(title='S&P500')
plt.show()

#separate plots for closed price and volume
sp500.loc['2012', ['Close','Volume']].plot(subplots=True)
plt.show()

# Plot the raw data before setting the datetime index
df.plot()
plt.show()

# Convert the 'Date' column into a collection of datetime objects: df.Date
df.Date = pd.to_datetime(df.Date)

# Set the index to be the converted 'Date' column
df.set_index('Date', inplace=True)

# Re-plot the DataFrame to see that the axis is now datetime aware!
df.plot()
plt.show()

# Plot the summer data
df.Temperature['2010-Jun':'2010-Aug'].plot()
plt.show()
plt.clf()

# Plot the one week data
df.Temperature['2010-06-10':'2010-06-17'].plot()
plt.show()
plt.clf()


# Import pandas
import pandas as pd

# Read in the data file: df
df = pd.read_csv(data_file)

# Print the output of df.head()
print(df.head())

# Read in the data file with header=None: df_headers
df_headers = pd.read_csv(data_file, header=None)

# Print the output of df_headers.head()
print(df_headers.head())

# Split on the comma to create a list: column_labels_list
column_labels_list = column_labels.split(',')

# Assign the new column labels to the DataFrame: df.columns
df.columns = column_labels_list

# Remove the appropriate columns: df_dropped
df_dropped = df.drop(list_to_drop, axis='columns')

# Print the output of df_dropped.head()
print(df_dropped.head())


# Convert the date column to string: df_dropped['date']
df_dropped['date'] = df_dropped['date'].astype(str)

# Pad leading zeros to the Time column: df_dropped['Time']
df_dropped['Time'] = df_dropped['Time'].apply(lambda x:'{:0>4}'.format(x))

# Concatenate the new date and Time columns: date_string
date_string = df_dropped['date'] + df_dropped['Time']

# Convert the date_string Series to datetime: date_times
date_times = pd.to_datetime(date_string, format='%Y%m%d%H%M')

# Set the index to be the new date_times container: df_clean
df_clean = df_dropped.set_index(date_times)

# Print the output of df_clean.head()
print(df_clean.head())


# Print the dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 0800':'2011-06-20 0900', 'dry_bulb_faren'])

# Convert the dry_bulb_faren column to numeric values: df_clean['dry_bulb_faren']
df_clean['dry_bulb_faren'] = pd.to_numeric(df_clean['dry_bulb_faren'], errors='coerce')

# Print the transformed dry_bulb_faren temperature between 8 AM and 9 AM on June 20, 2011
print(df_clean.loc['2011-06-20 0800':'2011-06-20 0900', 'dry_bulb_faren'])

# Convert the wind_speed and dew_point_faren columns to numeric values
df_clean['wind_speed'] = pd.to_numeric(df_clean['wind_speed'], errors='coerce')
df_clean['dew_point_faren'] = pd.to_numeric(df_clean['dew_point_faren'], errors='coerce')

# Print the median of the dry_bulb_faren column
print(df_clean['dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the time range '2011-Apr':'2011-Jun'
print(df_clean.loc['2011-Apr':'2011-Jun', 'dry_bulb_faren'].median())

# Print the median of the dry_bulb_faren column for the month of January
print(df_clean.loc['2011-Jan', 'dry_bulb_faren'].median())

# Downsample df_clean by day and aggregate by mean: daily_mean_2011
daily_mean_2011 = df_clean.resample('D').mean()

# Extract the dry_bulb_faren column from daily_mean_2011 using .values: daily_temp_2011
daily_temp_2011 = daily_mean_2011['dry_bulb_faren'].values

# Downsample df_climate by day and aggregate by mean: daily_climate
daily_climate = df_climate.resample('D').mean()

# Extract the Temperature column from daily_climate using .reset_index(): daily_temp_climate
daily_temp_climate = daily_climate.reset_index()['Temperature']

# Compute the difference between the two arrays and print the mean difference
difference = daily_temp_2011 - daily_temp_climate
print(difference.mean())

# From previous steps
is_sky_clear = df_clean['sky_condition']=='CLR'
sunny = df_clean.loc[is_sky_clear]
sunny_daily_max = sunny.resample('D').max()
is_sky_overcast = df_clean['sky_condition'].str.contains('OVC')
overcast = df_clean.loc[is_sky_overcast]
overcast_daily_max = overcast.resample('D').max()

# Calculate the mean of sunny_daily_max
sunny_daily_max_mean = sunny_daily_max.mean()

# Calculate the mean of overcast_daily_max
overcast_daily_max_mean = overcast_daily_max.mean()

# Print the difference (sunny minus overcast)
print(sunny_daily_max_mean - overcast_daily_max_mean)

#More data visualization using line plots
import matplotlib.pyplot as plt
climate2010.Temperature['2010-07'].plot()
plt.title('Temperature (July 2010)')
plt.show()

#Can plot histograms
climate2010['DewPoint'].plot(kind= 'hist', bins=30)
plt.title('Dew point distribution (2010)')
plt.show()

#Blox plot pandas
climate2010['DewPoint'].plot(kind='box')
plt.title('Dew Point distribution (2010)')
plt.show()

#subplots
climate2010.plot(kind='hist', normed=True, subplots=True)
plt.show()

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Select the visibility and dry_bulb_faren columns and resample them: weekly_mean
weekly_mean = df_clean[['visibility','dry_bulb_faren']].resample('W').mean()

# Print the output of weekly_mean.corr()
print(weekly_mean.corr())

# Plot weekly_mean with subplots=True
weekly_mean.plot(subplots=True)
plt.show()

# From previous steps
is_sky_clear = df_clean['sky_condition'] == 'CLR'
resampled = is_sky_clear.resample('D')
sunny_hours = resampled.sum()
total_hours = resampled.count()
sunny_fraction = sunny_hours / total_hours

# Make a box plot of sunny_fraction
sunny_fraction.plot(kind='box')
plt.show()

# Resample dew_point_faren and dry_bulb_faren by Month, aggregating the maximum values: monthly_max
monthly_max = df_clean[['dew_point_faren','dry_bulb_faren']].resample('M').max()

# Generate a histogram with bins=8, alpha=0.5, subplots=True
monthly_max.plot(kind='box')

# Show the plot
plt.show()

# Extract the maximum temperature in August 2010 from df_climate: august_max
august_max = df_climate.loc['2010-Aug', 'Temperature'].max()
print(august_max)

# Resample August 2011 temps in df_clean by day & aggregate the max value: august_2011
august_2011 = df_clean.loc['2011-Aug', 'dry_bulb_faren'].resample('D').max()

# Filter for days in august_2011 where the value exceeds august_max: august_2011_high
august_2011_high = august_2011.loc[august_2011 > august_max]

# Construct a CDF of august_2011_high
august_2011_high.plot(kind='hist', normed=True, cumulative=True, bins=25)

# Display the plot
plt.show()