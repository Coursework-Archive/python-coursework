'''
Timeseries kinds and applications 
Timeseries means data that changes over time 
This can take many different forms such as atmospheric CO2 over time,
the waveform of my voice as I am speaking. The fluctuation of a stock's
value over the year, or demographic information about a city. 

Time series consists of at least two things: 
    i. an array of numbers that represents data itself 
    ii. another array that contains a timestamp for each datapoint 

Plotting a pandas timeseries 
import matplotlib.pyplot as plt 
fig, ax = plt.subplots(figsize=(12, 6))
data.plot('date', 'close', ax=ax)
ax.set(title="AAPL daily closing price")

A machine learning pipeline 
* Feature extraction
* Model fitting 
* Prediction and validation 
'''
###Plotting a timeseries (I)
# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(y='data_values', ax=axs[0])
data2.iloc[:1000].plot(y='data_values', ax=axs[1])
plt.show()

###Plotting a timeseries (II)
# Plot the time series in each dataset
fig, axs = plt.subplots(2, 1, figsize=(5, 10))
data.iloc[:1000].plot(x="time", y="data_values", ax=axs[0])
data2.iloc[:1000].plot(x="time", y="data_values", ax=axs[1])
plt.show()

'''
Machine learning basics

Preparing data for scikit-learn
scikit-learn expects a particular structure of data

(samples, features)

Make sure that your data is at  least two-dimensional

Make sure the first dimension is samples

If your data is not in this shape (samples, featues) there are a few  options
for reshaping it. The most common approach is to "transpose" your data. This will
swap the first and last axis. This is most useful when your data is two diensional

array.T.shape


Another option is to use the dot-reshape method, which lets you specify the shape
you want

#if we're missing an axis, use .reshape():

array.shape

Output: (10,0)

array.reshape([-1, 1]).shape

Output: (10,1)

# -1 will automatically fill that axis with remaining values



Fitting a model with scikit-learn

# Import a support vector classifier 
from sklearn.svm import LinearSVC

# Instatiate this model
model = LinearSVC()

# Fit the model on some data
model.fit(X, y)

It is common for y to be of shape (samples, 1)


Investigating the model

# There is one coefficien per input features
model.coef_

Predicting with a fit model 
# Generate predictions 
predictions = model.predict(X_test)
'''

###Fitting a simple model: classification 
from sklearn.svm import LinearSVC

# Construct data for the model
X = data[['petal length (cm)', 'petal width (cm)']]
y = data[['target']]

# Fit the model
model = LinearSVC()
model.fit(X, y)

###Predicting and using the classification model 
# Create input array
X_predict = targets[['petal length (cm)', 'petal width (cm)']]

# Predict with the model
predictions = model.predict(X_predict)
print(predictions)

# Visualize predictions and actual values
plt.scatter(X_predict['petal length (cm)'], X_predict['petal width (cm)'],
            c=predictions, cmap=plt.cm.coolwarm)
plt.title("Predicted class values")
plt.show()


###Fitting a simple model regression 
from sklearn import linear_model

# Prepare input and output DataFrames
X = boston[["AGE"]]
y = boston[["RM"]]

# Fit the model
model = linear_model.LinearRegression()
model.fit(X,y)


###Predicting using the regression model 
# Generate predictions with the model using those inputs
predictions = model.predict(new_inputs.reshape([-1, 1]))

# Visualize the inputs and predicted values
plt.scatter(new_inputs, predictions, color='r', s=3)
plt.xlabel('inputs')
plt.ylabel('predictions')
plt.show()


'''
Getting to know our data 
Loading auditory data (heartbeats)

from glob import glob 
files = glob('data/heartbeat-sounds/files/*.wav')

print(files)

import librosa as lr 
# 'load' accepts a path to an audio file 
audio, sfreq = lr.load('data/heartbeat-sounds/proc/files/murmur__201101051104.wav')

prints(sfreq)

Inferring time from samples 
* If we know the sampling rate of a timeseries, then we know the timestamp of each datapoint 
relative to the first datapoint 
* Note: this assumes the sampling rate is fixed and no data points are lost 

Creating a time array (I)
* Create an array of indices, one for each sample, and divide by the sampling frequency 
indices = np.arange(0, len(udio))
time = indices / sfreq


Creating a time array (II)
* Find the time stamp for the N-1th data point. Then use linspace() to interpolate from zero 
to that time 

final_time = (len(audio) - 1) / sfreq
time = np.linspace(0, final_time, sfreq)
'''

###Inspecting the classification data 
import librosa as lr
from glob import glob

# List all the wav files in the folder
audio_files = glob(data_dir + '/*.wav')

# Read in the first audio file, create the time array
audio, sfreq = lr.load(audio_files[0])
time = np.arange(0, len(audio)) / sfreq

# Plot audio over time
fig, ax = plt.subplots()
ax.plot(time, audio)
ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
plt.show()

###Inspecting the regression data 
# Read in the data
data = pd.read_csv('prices.csv', index_col=0)

# Convert the index of the DataFrame to datetime
data.index = pd.to_datetime(data.index)
print(data.head())

# Loop through each column, plot its values over time
fig, ax = plt.subplots()
for column in data:
    data[column].plot(ax=ax, label=column)
ax.legend()
plt.show()

'''
Classification and feature engineering 

To plot raw audio, we need two things: the raw audio waveform, usually in a 
1- or 2- dimensional array. We also need the time point of each sample 

We can calculate the time by dividing the index of each sample by the sampling
frequency of the timeseries. This gives us the time for each sample relative to
the beginning of the audio 

ixs = np.arange(audio.shape[-1])
time = ixs / sfreq
fig, ax = plt.subplots()
ax, plot(time, audio)


Using the raw data as input to a classifier is usually too noisy to be useful
An easy first step is to calculate summary statistics of our data, which is 
removes the "time" dimension and gives us a more traditional classification

What features to use?
* Using raw timeseries data is too noisy for classification 
* We need to calcullate features!
* An easy start:summarize your audio data 

Here we show how to calculated multiple features for several timeseries 
 print(audio.shape)
# (n_files, time)

Output: (20, 7000)

#Collapsing 1-D array to 2-D array 
means = np.mean(audio, axis=-1) #axis = -1 keyword - collapses across the las dimension, time
maxs = np.max(audio, axis=-1)
stds = np.std(audio, axis=-1)

print(means.shape)
# (n_files,)

Output: (20,)


Fitting a classifier with scikit-learn 
* We've just collapsed a 2-D dataset (samples x time) into several features of a 1-D 
dataset (samples)
* We can combine each feature, and use it as an input to a model 
* If we have a label for each sample, we can use scikit-learn to create and fit a classifier 

 Preparing you feature for scikit-learn
 # Import linear classifier 
 from sklearn.svm import LinearSVC
 
 #Note that means are reshaped to wirk with scikit-learn 
 X = np.column_stack([means, maxs, stds])
 y = labels.reshape([-1,1])
 model = LinearSVC()
 model.fit(X, y)
 
 from sklearn.metrics import accuracy_score
 
 # Different input data
 predictions = model.predict(X_test)
 
 # Score our model with % correct
 # Manually
 percent_score = sum(predictions == labels_test / len(labels_test))
 
 # Using a sklearn scorer
 percent_score = accuracy_score(labels_test, predictions)
 '''
 
 ###Many repetitions of sounds 
 fig, axs = plt.subplots(3, 2, figsize=(15, 7), sharex=True, sharey=True)

# Calculate the time array
time = np.arange(normal.shape[0]) / sfreq

# Stack the normal/abnormal audio so you can loop and plot
stacked_audio = np.hstack([normal, abnormal]).T

# Loop through each audio file / ax object and plot
# .T.ravel() transposes the array, then unravels it into a 1-D vector for looping
for iaudio, ax in zip(stacked_audio, axs.T.ravel()):
    ax.plot(time, iaudio)
show_plot_and_make_titles()


###Invariance in time 
# Average across the audio files of each DataFrame
mean_normal = np.mean(normal, axis=1)
mean_abnormal = np.mean(abnormal, axis=1)

# Plot each average over time
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
ax1.plot(time, mean_normal)
ax1.set(title="Normal Data")
ax2.plot(time, mean_abnormal)
ax2.set(title="Abnormal Data")
plt.show()


###Build a classification model
from sklearn.svm import LinearSVC

# Initialize and fit the model
model = LinearSVC()
model.fit(X_train, y_train)

# Generate predictions and score them manually
predictions = model.predict(X_test)
print(sum(predictions == y_test.squeeze()) / len(y_test))

'''
Improving the deature we use for classification 
The auditory envelope 
* Smooth the data to calculate the auditory envelope 
* Related to the total amount of audio energy present at each moment of time 

Smoothing over time 
* instead of averaging over all time, we can do a local average 
* This is called smoothing your timeseries 
* It removes short-term noise, while retaining the gerneral pattern 

Calculating a rolling window statistic 
# Audio is a Pandas DataFrame 
print(audio.shape)
# (n_times, n_audio_files)

# Smooth our data by taking the rolling mean in a window of 50 samples 
window_size = 50
windowed = audio.rolling(window=window_size)
audio_smooth = windowed.mean()

Calculating the auditory envelope 
* First rectify your audio, then smooth it 
audio_rectified = audio.apply(np.abs)
audio_envelope = audio_rectified.rolling(50).mean()

Feature engineering the envelope 
# Calculate several features of the envelope, one per sound 
envelope_mean = np.mean(audio_envelope, axis=0)
envelope_std = np.std(audio_envelope, axis=0)
envelope_max = np.max(audio_envelope, avis=0)

# Create our training data for a classifier
X = np.column_stack([envelope_mean, envelope_std, envelope_max]) #envelope mean, envelope std, and envelope max are all different features 
y = labels.reshape([-1, 1])

Cross validation for classification 
* cross_val_score automates the process of:
    - Splitting data into training / validation sets 
    - Fitting the model on training data 
    - Scoring it on validation data 
    - Repeating this process 

Using cross_val_score 
from sklearnmodel_selection import cross_val_score 

model = LinearSVC()
scores = cross_val_score(model, X, y, cv=3)
print(scores)

Auditory features: The Tempogram 
* We can summarize more complex temporal information with timeseries-specific functions
* librosa is a great library for auditory and timeseries feature engineering 
* Here we'll calculalte the tempogram, which estimates the tempo of a sound over time 
* We can calculalte summary statistics of tempo in the same way that we can for the envelope

Computing the tempogram 
# Import librosa and calculate the tempo of a 1-D sound array 
import librosa as lr 
audio_tempo = lr.beat.tempo(audio, sr=sfreq, hop_legth=2**6, aggregate=None)
'''

###Calculating the envelope of sound 
# Plot the raw data first
audio.plot(figsize=(10, 5))
plt.show()

# Rectify the audio signal
audio_rectified = audio.apply(np.abs)

# Plot the result
audio_rectified.plot(figsize=(10, 5))
plt.show()

# Smooth by applying a rolling mean
audio_rectified_smooth = audio_rectified.rolling(50).mean()

# Plot the result
audio_rectified_smooth.plot(figsize=(10, 5))
plt.show()


###Calculalting the features from the envelopes
# Calculate stats
means = np.mean(audio_rectified_smooth, axis=0)
stds = np.std(audio_rectified_smooth, axis=0)
maxs = np.max(audio_rectified_smooth, axis=0)

# Create the X and y arrays
X = np.column_stack([means, stds, maxs])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
from sklearn.model_selection import cross_val_score
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))


### Derivative Features 
# Calculate the tempo of the sounds
tempos = []
for col, i_audio in audio.items():
    tempos.append(lr.beat.tempo(i_audio.values, sr=sfreq, hop_length=2**6, aggregate=None))

# Convert the list to an array so you can manipulate it more easily
tempos = np.array(tempos)

# Calculate statistics of each tempogram
tempos_mean = tempos.mean(axis=-1)
tempos_std = tempos.std(axis=-1)
tempos_max = tempos.max(axis=-1)

### Derivative features: The tempoogram 
# Create the X and y arrays
X = np.column_stack([means, stds, maxs, tempos_mean, tempos_std, tempos_max])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))

'''
The spectrogram - spectral changes to sound over time 

Fourier transfors 
* Timeseries data can be described as a combination of quickly-changing things and slowly 
changing things 
* At each moment in time, we can describe the relative presence of fast and slow moving 
components 
* The simplest way to do this is called Fourier Transform 
* This converts a songle timeseries into an array that describes the timeseries as  a combination
of oscillations 

Spectograms: combinations of windows Fourier transforms 
* A spectrogram is a collection of windowed Fourier transforms over time 
* Similar to how a rolling mean was calculated:
    1. Choose a window size and shape 
    2. At a timepoint, calculate the FFT for that window 
    3. Slide the window oer by one 
    4. Aggregate the results 
* Called a Short-Time Fouorier Transform (STFT)

Calculating the STFT 
* We can calculate the STFT with librosa 
* There are several parameters we can tweak (such as window size)
* For our purposes, we'll convert into decibels which normalize the average values of all frequencies 
* We can then visualize it with the specshow() function 


Calculalting the STFT with corde 
# Import the functions we'll use for the STFT
from librosa.core import stft, aplitude_to_db
from librosa.display import specshow

# Calculalte our STFT
HOP_LENGTH = 2**4
SIZE_WINDOW = 2**7
audio_spec = stft(audio, hop_legth=HOP_LENGTH, n_fft=SIZE_WINDOW)

# Convert into decivels for visualization 
spec_db = aplitude_to_db(audio_spec)

# Visualize
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_legth=HOP_LENGTH)

Spectral feature engineering 
* Each timeseries has a different spectral pattern
* We can calculalte these spectral patterns by analyzing the spectrogram 
* For example, spectral bandwidth and spectral centroids describe where most of the energy
is at each moment in time 

Calculalting spectral features 
# Calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]

# Display these features on top of thespectrogram
ax = specshow(spec, x_axis='time', y_axis='hz', hop_legth=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=0.5) 

Combining spectral and temporal features in a classifier 
centroids_all = []
bandwidths_all = []
for spec in spectrograms:
    bandwidths = lr.feature.spectral_bandwidth(S=lr.db_to_amplitude(spec))
    centroids = lr.feature.spectral_centroid(S=lr.db_to_aplitude(spec))
    # Calculate the mean spectral bandwidth
    bandwidths_all.append(np.mean(bandwidths))
    # Calculate the mean spectral centroid 
    centroids_all.append(np.mean(centroids))

# Create our X matrix
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, abndwidths_all, centroids_all])
'''


###Spectrograms of heartbeat audio
# Import the stft function
from librosa.core import stft

# Prepare the STFT
HOP_LENGTH = 2**4
spec = stft(audio, hop_length=HOP_LENGTH, n_fft=2**7)


from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert into decibels
spec_db = amplitude_to_db(spec)

# Compare the raw audio to the spectrogram of the audio
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
axs[0].plot(time, audio)
specshow(spec_db, sr=sfreq, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
plt.show()


###Engineering spectral features 
import librosa as lr

# Calculate the spectral centroid and bandwidth for the spectrogram
bandwidths = lr.feature.spectral_bandwidth(S=spec)[0]
centroids = lr.feature.spectral_centroid(S=spec)[0]


from librosa.core import amplitude_to_db
from librosa.display import specshow

# Convert spectrogram to decibels for visualization
spec_db = amplitude_to_db(spec)

# Display these features on top of the spectrogram
fig, ax = plt.subplots(figsize=(10, 5))
ax = specshow(spec_db, x_axis='time', y_axis='hz', hop_length=HOP_LENGTH)
ax.plot(times_spec, centroids)
ax.fill_between(times_spec, centroids - bandwidths / 2, centroids + bandwidths / 2, alpha=.5)
ax.set(ylim=[None, 6000])
plt.show()



###Combining many features in a classifier 
# Loop through each spectrogram
bandwidths = []
centroids = []

for spec in spectrograms:
    # Calculate the mean spectral bandwidth
    this_mean_bandwidth = np.mean(lr.feature.spectral_bandwidth(S=spec))
    # Calculate the mean spectral centroid
    this_mean_centroid = np.mean(lr.feature.spectral_centroid(S=spec))
    # Collect the values
    bandwidths.append(this_mean_bandwidth)  
    centroids.append(this_mean_centroid)

# Create X and y arrays
X = np.column_stack([means, stds, maxs, tempo_mean, tempo_max, tempo_std, bandwidths, centroids])
y = labels.reshape([-1, 1])

# Fit the model and score on testing data
percent_score = cross_val_score(model, X, y, cv=5)
print(np.mean(percent_score))


'''
Predicting data over time 
The biggest difference between regressio nand classification is that regression models 
predict continuous outputs. Whereas, classification models predict catagorical outputs.
Both Regressio nand correlation reflect the extent to which the values of the two variables 
have a consistent relationship (either they both go down or up together, or they have an 
inverse relationship). Regression resultls in the model of the data while correlation
is just a single statistic that describes the data. Regression models have more information
about the data, while correlation is easier to calculalte the interpret 

Classification vs. Regression 
Classification                              Regression  
classification_model.predict(X_test)        regression_model.predict(X_test)

Correlation and regression 
* Regression is similar to calculalting correlation, with some key differences 
    -Regression: A process that resuts in a formal model of the data 
    -Correlation: A statistic that describes the data. Less information than regression
    model.
    
When running regression models with timeseries data, it's important to visualize how the data
changes over time. You can either do this by plotting the whole timeseries at once, or by 
directly comparing two segments of time 

Correlaation between variables often changes over time  
* Timeseries often have patterns that change over time 
* Teo timeseries that seem correlated at one moet may not remain so over time 

Visualizing relationships between tiemseries 
fig, ax = plt.subplots(1, 2)

# Make a line plot for each timeseries 
axs[0].plot(x, c='k', lw=3, alpha=.2)
axs[0].plot(y)
axs[0].set(xlabel='time', title='X values = time')

# Encode time as color in a scatterplot
axs[1].scatter(x_long, y_long, c=np.arange(len(x_long)), cmap='viridis')
axs[1].set(xlabel='x', ylabel='y', title='Color = time')


Regression models with scikit-learn 
from sklearn.linear_model import LinearRegression 
model = LinearRegression()
model.fit(X, y)
model.predict(X)

Visualize predictions with scikit-learn
alphas = [.1, 1e2, 1e3]
ax.plot(y_test, color='k', alpha=.3, lw=3)
ax.plot(y_test, color='k', alpha=.3, lw=3)

for ii, alpha in enumerate(alphas):
    y_predicted = Ridge(alpha=alpha).fit(X_train, y_train).predict(X_test)
    ax.plot(y_predicted, c=cmap(ii / len(alphas)))
ax.legend(['True values', 'Model 1', 'Model 2', 'Model 3'])
ax.set(xlabel="Time")

Scorng regression models 
* Two most common methods:
    -Correlation(r)
    -Coefficient of determination (R^2)

Coefficient of Determination (R^2)
* The value of R^2 is bounded on the top by 1 and can be infinitely low
* Values closer to 1 mean the model does a better job of predicting outputs 

1 - (error(model)/variance(testdata))

R^2 in scikit-learn 

from sklearn.metrics import r2_score
print(r2_score(y_predicted, y_test))
'''

###Introducing the dataset 
# Plot the raw values over time
prices.plot()
plt.show()

# Scatterplot with color relating to time
prices.plot.scatter('EBAY', 'YHOO', c=prices.index, 
                    cmap=plt.cm.viridis, colorbar=False)
plt.show()


###Fitting a simple regression model
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Use stock symbols to extract training data
X = all_prices[["EBAY", "NVDA", "YHOO"]]
y = all_prices[["AAPL"]]

# Fit and score the model with cross-validation
scores = cross_val_score(Ridge(), X, y, cv=3)
print(scores)

###Visualizing predicted values 
# Visualize our predictions along with the "true" values, and print the score
fig, ax = plt.subplots(figsize=(15, 5))
ax.plot(y_test, color='k', lw=3)
ax.plot(predictions, color='r', lw=2)
plt.show()


'''
Data is messy 
* real-world data is often messy 
* The two most common problems are missing data and outliers 
* This often happens because of human error, machine sensor malfunction, database failures, etc. 
* Visualizing your raw data makes it easier to spot these problems 

Fixing data that is missing - Interpolation
* A common way to deal with missing data is to interpolate missing values 
* With timeseries data, you can use time to assist in interpolation 
* In this case, interpolation means using the known values either side of a gap in the data to make 
assumptions about what's missing 

Iterpolation in Pandas 
# Return a boolean that notes where missing values are 
missing = prices.isna()

# Interpolate linearly within missing windows 
prices_interp = prices.interpolate('linear')

# Plot the interpolated data in red and the data w/ missing values in black 
ax = prices_interp.plot(c='r')
prices.plot(c='k', ax=ax, lw=2)

Using a rolling window to transform data 
* Another common use of rolling windows is to transform the data 
* We've already done this once, in order to smooth the data 
* However, we can also use this to do more complex transformations 

Transforming data to standardize variance 
* A common transformation to apply to data is to stadardize its mean and variance over time. There are many 
ways to do this 
* Here, we'll show how to convert your dataset so that each point represents the % change over a previous window
* The makes timepoints more comparable to one another if the absolute values of data change a lot 

Transforming to percent change with Pandas 
def percent_change(values):
    """Calculates the % change between the last value 
    and the mean of precious values"""
    # Separate the last values and all previous values into variables 
    previous_value = value[:-1]
    last_value = values[-1]
    
    # Calculate the % difference between the last value 
    # and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) \
    / np.mean(previouos_values)
    return precent_change

Applying this to our data 
# Plot the raw data
fig, axs = plt.subplots(1, 2, figsizes=(10, 5))
ax = prices.plot(ax=axs[0])

# Calculate % change and plot 
ax = prices.rolling(window=20).aggregate(percent_change).plot(ax=ax[1])
ax.legend_.set_visible(False)

Finding outliers is your data 
* Outliers are datapoints that are significantly statistically different from the dataset 
* They can have negative effects on the predictive power of your model, biasing it away
from its "true" value 
* One solution is to remove or replace outliers with a ore representative value 
    Be very careful about doing this - often it is difficulet to determine what is the 
    legitimately extreme value vs an abberation.


Plotting a threshold on out data 
fig, axs = plt.subplot(1, 2, figsize=(10, 5))
for data, ax in zip([prices, prices_perc_change], axs):
    # Calculalte the mean / standard deviation for the data 
    this_mean = data.mean()
    this_std = data.std()
    
    # Plot the data, with a window that is 3 standard deviations 
    # around the mean 
    data.plot(ax=ax)
    ax.axhline(this_mean + this_std * 3, ls='--', c='r')
    ax.axhline(this_mean - this_std * 3, ls='--', c='r')

Replacing outliers using the threshold 
#Center the data so the mean is 0
prices_outlier_centered = prices_outliers_perc - prices_outlier_perc.mean()

#Calculate standard deviation 
std = prices_outlier_perc.std()

#Use the absolute value of each datapoint 
#to make it easier to find outliers 
outliers = np.abs(prices_outlier_centered) > (std * 3)

# Replace outliers with the median value 
# We'll use np.nanmean since there may be nans around the outliers 
prices_outlier_fixed = prices_outlier_centered.copy()
prices_outlier_fixed[outliers] = np.nanmedian(prices_outlier_fixed)

Visualize the results 
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
prices_outlier_centered.plot(ax=axs[0])
prices_outlier_fixed.plot(ax=axs[1])
'''

###Visualizing messy data 
# Visualize the dataset
prices.plot(legend=False)
plt.tight_layout()
plt.show()

# Count the missing values of each time series
missing_values = prices.isnull().sum()
print(missing_values)

###Imputing mssing values 
# Create a function we'll use to interpolate and plot
def interpolate_and_plot(prices, interpolation):

    # Create a boolean mask for missing values
    missing_values = prices.isna()

    # Interpolate the missing values
    prices_interp = prices.interpolate('linear')

    # Plot the results, highlighting the interpolated values in black
    fig, ax = plt.subplots(figsize=(10, 5))
    prices_interp.plot(color='k', alpha=.6, ax=ax, legend=False)
    
    # Now plot the interpolated values on top in red
    prices_interp[missing_values].plot(ax=ax, color='r', lw=3, legend=False)
    plt.show()

# Interpolate using the latest non-missing value
interpolation_type = 'zero'
interpolate_and_plot(prices, interpolation_type)


# Interpolate linearly
interpolation_type = 'linear'
interpolate_and_plot(prices, interpolation_type)

# Interpolate with a quadratic function
interpolation_type = 'quadratic'
interpolate_and_plot(prices, interpolation_type)


###Transforming raw data 
# Your custom function
def percent_change(series):
    # Collect all *but* the last value of this window, then the final value
    previous_values = series[:-1]
    last_value = series[-1]

    # Calculate the % difference between the last value and the mean of earlier values
    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply your custom function and plot
prices_perc = prices.rolling(20).apply(percent_change)
prices_perc.loc["2014":"2015"].plot()
plt.show()


###Handling outliers
def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    absolute_differences_from_mean = np.abs(series - np.mean(series))
    
    # Calculate a mask for the differences that are > 3 standard deviations from zero
    this_mask = absolute_differences_from_mean > (np.std(series) * 3)
    
    # Replace these values with the median accross the data
    series[this_mask] = np.nanmedian(series)
    return series

# Apply your preprocessing function to the timeseries and plot the results
prices_perc = prices_perc.apply(replace_outliers)
prices_perc.loc["2014":"2015"].plot()
plt.show()


'''
Creating feature over time 
Using .aggregate for feature extraction 
# Visualize the raw data 
print(prices.head(3))

#Calculate a rolling window, then extract two features 
feats = prices.rolling(20).aggregate([np.std, np.max]).dropna()
print(feat.head(3))

Using partial() in Python
# If we just take the mean, it returns a single value 
a = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
print(np.mean(a))

# We can use the partial function to initialize np.mean
# with an axis parameter 

from functool import partial
mean_over_first_axis = partial(np.mean, axis=0)

print(mean_over_first_axis(a))

Percentiles summarize your data 
* Percentiles are a useful way to get more fine-grained summaries of your data 
(as opposed to using np.mean)
* For a given dataset, the Nth percentile is the value where N% of the data is 
below that datapoint, and 100-N% of the data is above that datapoint 

print(np.percentile(np.linspace(0, 200), q=20))

Combining np.percentile() with partial functions to calculate a range of percentiles 

data = np.linspace(0m 100)

# Create a list of functions using a list comprehension
percentile_funcs = [partial(np.percentile, q=ii) for ii in [20, 40, 60]]

# Calculate the output of each function in the same way 
percentiles = [i_func(data) for i_func in percentile_funcs]
print(percentiles)

# Calculate multiple percentiles of a rilling window 
data.rolling(20).aggregate(percentiles)

Calculating "date-based" features
*Thus far we've focused on calculating "statistical" features - these are features that 
correspond statistical properties of the data, link "mean", "standard deviation", etc
* However, don't forget that timeseries data often has more "human" features associate 
it, like days of the week, holidays, etc. 
* These features are often useful when dealing with timeseries data that spans multiple
years (such as stock values over time) 


datetime features using Pandas 
# Ensure our index is datetime 
prices.index = pd.to_datetime(prices.index)

# Extract datetime features 
day_of_week_num = prices.index.weekday
print(day_of_week_num[:10])

day_of_week = prices.index.weekday_name
print(day_of_week[:10])
'''

###Engineering multiple rolling features at once 
# Define a rolling window with Pandas, excluding the right-most datapoint of the window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')

# Define the features you'll calculate for each window
features_to_calculate = [np.min, np.max, np.mean, np.std]

# Calculate these features for your rolling window object
features = prices_perc_rolling.aggregate(features_to_calculate)

# Plot the results
ax = features.loc[:"2011-01"].plot()
prices_perc.loc[:"2011-01"].plot(ax=ax, color='k', alpha=.2, lw=3)
ax.legend(loc=(1.01, .6))
plt.show()


###Percentiles and partial functions 
# Import partial from functools
from functools import partial
percentiles = [1, 10, 25, 50, 75, 90, 99]

# Use a list comprehension to create a partial function for each quantile
percentile_functions = [partial(np.percentile, q=percentile) for percentile in percentiles]

# Calculate each of these quantiles on the data using a rolling window
prices_perc_rolling = prices_perc.rolling(20, min_periods=5, closed='right')
features_percentiles = prices_perc_rolling.aggregate(percentile_functions)

# Plot a subset of the result
ax = features_percentiles.loc[:"2011-01"].plot(cmap=plt.cm.viridis)
ax.legend(percentiles, loc=(1.01, .5))
plt.show()

###Using "date" information
# Extract date features from the data, add them as columns
prices_perc['day_of_week'] = prices_perc.index.dayofweek
prices_perc['week_of_year'] = prices_perc.index.weekofyear
prices_perc['month_of_year'] = prices_perc.index.month

# Print prices_perc
print(prices_perc)

'''
Time-delayed features and auto-regressive models
The past is useful
* Timeseries data almost always have information that is shared between timepoints 
* Information in the past can help predict what happens in the future 
* Often the features best-suited to rpedict a timeseries are previous values of 
the same series  

A note on smoothness and auto-correlation 
* A common question to ask of a timeseries: how smooth is the data 
* AKA, how correlated is a timepoint with itse neighboring timepoints (called autocorrelation)
* The amount of auto-correlation in data will impact your models 

Creating time-lagged features 
* Let's see how we could build a model that uses values in the past as input features.
* We can use this to assess how auto-correlation our signal is (and lot of other stuff too)

Time-shifting data with Pandas
print(df)

# Shift a DataFrame/Series by 3 index values towards the past 
print(df.shift(3))


Creating a time-shifted DataFrame
# data is a pandas Series containing time series data
data = pd.Series(. . .)

# Shifts
shifts = [0, 1, 2, 3, 4, 5, 6, 7]

# Create a dictionary of time-shifted data
many_shifts = {'lag_{}'.format(ii): data.shift(ii) for ii in shifts}

# Convert them into a dataframe
many_shifts = pd.DataFrame(many_shifts)

Fitting a modelwith time-shifted features 
# Fit the model using these input features 
model = Ridge()
model.fit(many_shifts, data)

Interpreting the auto-regressive model coefficients 
# Visualize the fit model coefficients 
fig, ax = plt.subplots()
ax.bar(many_shifts.columns, model.coef_)
ax.set(xlabel='Coefficient name', ylabel='Coefficient value')

# Set formatting so it look nice
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
'''

###Creating time-shifted features 
# These are the "time lags"
shifts = np.arange(1, 11).astype(int)

# Use a dictionary comprehension to create name: value pairs, one pair per shift
shifted_data = {"lag_{}_day".format(day_shift): prices_perc.shift(day_shift) for day_shift in shifts}

# Convert into a DataFrame for subsequent use
prices_perc_shifted = pd.DataFrame(shifted_data)

# Plot the first 100 samples of each
ax = prices_perc_shifted.iloc[:100].plot(cmap=plt.cm.viridis)
prices_perc.iloc[:100].plot(color='r', lw=2)
ax.legend(loc='best')
plt.show()


###Special case: Auto-regressive models
# Replace missing values with the median for each column
X = prices_perc_shifted.fillna(np.nanmedian(prices_perc_shifted))
y = prices_perc.fillna(np.nanmedian(prices_perc))

# Fit the model
model = Ridge()
model.fit(X, y)

###Visualize regression coeficients 
def visualize_coefficients(coefs, names, ax):
    # Make a bar plot for the coefficients, including their names on the x-axis
    ax.bar(names, coefs)
    ax.set(xlabel='Coefficient name', ylabel='Coefficient value')
    
    # Set formatting so it looks nice
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    return ax

###Visualize regression coefficients 
# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()

###Auto-regression with a smoother time series 
# Visualize the output data up to "2011-01"
fig, axs = plt.subplots(2, 1, figsize=(10, 5))
y.loc[:'2011-01'].plot(ax=axs[0])

# Run the function to visualize model's coefficients
visualize_coefficients(model.coef_, prices_perc_shifted.columns, ax=axs[1])
plt.show()


'''
Cross-validation timeseries data

Cross validation with scikit-learn
# Iterating oer the "split" method yields train/test indices 
for tr, tt in cv.split(X, y):
    model.fit(X[tr], y[tr])
    model.score(X[tt], y[tt])

Cross validation types: KFold 
* KFold cross-validation splits your data into multiple "folds" of equal size 
* It is one of the mose common cross-validation routines 
    from sklearn.model_selection import KFold 
    cv = KFold(n_splits=5)
    for tr, tt in cv.split(X, y):
        ...

fig, axs = plt.subplots(2, 1)

# Plot the indices chosen for validation on each loop 
axs[0].scatter(tt, [0] * len(tt), marker='-', s=2, lw=40)
axs[0].set(ylim=[-.1, .1], title='Test set indices (color=CV loop)',
            xlabel='Inde of ra data')

# Plot the model predictions on each iteration
ax[1].plot(model.redict(X[tt]))
ax[1].set(title='Test set predictions on each CV loop',
            xlabel='Prediction index')

A note on shuffling your data 
Many CV iterators let you shuffle data as a part of the cross-valiation process
This only works if the data is i.i.d., which timeries usually is not 
This should not shuffle your data when making predictions with timeseries 
    from sklearn.model_selection import ShuffleSplit 
    
    cv = ShuffleSplit(n_splits=3)
    for tr, tt in cv.split(X, y):
        ...

Using the time series CV iterator
* Thus far, we've broken the linear passage of time in the cross validation 
* However, you generally hould not use datapoints in the future to predict data in the 
past 
* One approach: Always use training data from the past to predict the future

Visualizing time seires cross validation iterators 
# Import and initializing the cross-validation iterator 
from sklearn.model_selection import TimeSeriesSplit
cv = TimeSeriesSplit(n_splits=10)

fig, ax = plt.subplots(figsize=(10, 5))
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot training and test indices 
    l1 = ax.scatter(tr, [ii] * len(tr), c=[plt.cm.coolwarm(.1)],
                marker='_', lw=6)
    l2 = ax.scatter(tt, [ii] * len(tt), c=[plt.cm.coolwarm(.9)],
                    marker='_', lw=6)
    ax.set(ylim=[10, -1], title='TimeSeriesSplit behavior', 
            xlabel='data index', ylabel='CV iteration')
    ax.legend([l1, l2], ['Training', 'Validation'])
    
Custom scoring functions in scikit-learn
def myfunction(estimator, X, y):
    y_pred = estimator.predict(X)
    my_custom_score = my_custom_function(y_pred, y)
    return my_custom_score 

A custom correlation function for scikit-learn
def my_personr(est, X, y):
    # Generate predictions and convert to a vector 
    y_pred = est.predict(X).squeeze()
    
    # Use the numpy "corrcoef" function to calculate a correlation matrix
    my_corrcoef_matrix = np.corrcoef(y_pred, y.squeeze())
    
    # Return a single correlation value from the matrix
    my_corrcoef = my_corrcoef[1, 0]
    return my_corrcoef
'''


###Cross-validation without shuffeling 
# Import ShuffleSplit and create the cross-validation object
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data, score the predictions, and collect
    prediction = model.predict(X[tt])
    score = r2_score(y[tt], prediction)
    results.append((prediction, score, tt))

# Custom function to quickly visualize predictions
visualize_predictions(results)


###Cross-validation without shuffling 
# Create KFold cross-validation object
from sklearn.model_selection import KFold
cv = KFold(n_splits=10, shuffle=False, random_state=1)

# Iterate through CV splits
results = []
for tr, tt in cv.split(X, y):
    # Fit the model on training data
    model.fit(X[tr], y[tr])
    
    # Generate predictions on the test data and collect
    prediction = model.predict(X[tt])
    results.append((prediction, tt))
    
# Custom function to quickly visualize predictions
visualize_predictions(results)


###Time-based cross-validation 
# Import TimeSeriesSplit
from sklearn.model_selection import TimeSeriesSplit 

# Create time-series cross-validation object
cv = TimeSeriesSplit(n_splits=10)

# Iterate through CV splits
fig, ax = plt.subplots()
for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Plot the training data on each iteration, to see the behavior of the CV
    ax.plot(tr, ii + y[tr])

ax.set(title='Training data on each CV iteration', ylabel='CV iteration')
plt.show()


'''
Stationarity 
* Stationary time series do not change their statistical properties over time 
* E.g., mean, standard deviation, trends 
* Most time series are non-stationary to some extent 

Model stability 
* Non-stationary data resutls in variability in out model 
* The statistical properties the model finds may change with the data 
* In addition, we will be less certain about the correct value of model parameters 
* How can we quantify this?

Cross validation to quanify parameter stability 
* One appraoch: use cross-validation 
* Calculate model parameters on each iteration
* Assess parameter stability across all CV splits 

Bootstrapping the mean 
* Bootstrapping is a common way to assess variability 
* The bootstrap:
1. Take a random sample of data with replacment 
2. Calculate the mean of the sample
3 Repeat this process many times (1000s)
4. Calculate the percentiles of the result (usually 2.5, 97.5)
The result is a 95% confidence interval of the mean of each coefficient 

from sklearn.utils impoort resample 

# cv_coefficients has shape (n_cv_folds, n_coefficients)
n_boots = 100
bootstrap_means = np.zeros(boots, n_coefficients)
for ii in range(n_boots):
    # Generate randm indices for our data with replacement, 
    #then take the sample mean 
    random_sample = resample(cv_coefficients)
    bootstrap_means[ii] = random_sample.mean(axis=0)

# compute the percentiles of choice for the boostrapped means 
percentiles = np.percentile(bootstrap_means, (2.5, 97.5), axis=0)

fig, ax = plt.subplots()
ax.scatter(many_shifts.columns, percentiles[0], marker='_', s=200)
ax.scatter(many_shifts.columns, percentiles[1], marker='_', s=200)

Assessing model performance stability 
* If using the TimeSeriesSplit, can plot the model's score over time. 
* This is useful in finding certain regions of time that hurt the score 
* Also useful to find non-stationary signals 

Model performance over time 
def my_corrcoef(est, X, y):
    """Return the correlation coefficient 
    between model predictions and a validation set."""
    return np.corrcoef(y, ext.predict(X))[1, 0]
    
# Grab the date of the firt index of each validation set
first_indices = [data.index[ttt[0]] for tr, tt in cv.split(X, y)]

# Calculate the CV scores and convert to Pandas Series 
cv_scores = cross_val_score(model, X, y, cv=cv, scoring=my_corrcoef)
cv_scores = pb.Series(cv_scores, index=first_indices)  

fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

# Calculate a rolling mean of scores over time 
cv_scores_mean = cv_scores.rolling(10, min_perios=1).mean()
cv_scores.plot(ax=ax[0])
ax[0].set(title='Validation scores (correlation)', ylim=[0, 1])

# Plot the raw data 
data.plot(ax=axs[1])
axs[1].set(title='Validation data')

Fixed windows with time series cross-validation
# Only keep the last 100 datapoints in the training data 
window = 100

# Initialize the CV with this window size 
cv = TimeSeriesSplit(n_splits=10, max_train_size=window)
'''


###Botstrapping a confidence interval
from sklearn.utils import resample

def bootstrap_interval(data, percentiles=(2.5, 97.5), n_boots=100):
    """Bootstrap a confidence interval for the mean of columns of a 2-D dataset."""
    # Create our empty array to fill the results
    bootstrap_means = np.zeros([n_boots, data.shape[-1]])
    for ii in range(n_boots):
        # Generate random indices for our data *with* replacement, then take the sample mean
        random_sample = resample(data)
        bootstrap_means[ii] = random_sample.mean(axis=0)
        
    # Compute the percentiles of choice for the bootstrapped means
    percentiles = np.percentile(bootstrap_means, percentiles, axis=0)
    return percentiles
    


###Calculating variability in model coefficients
# Iterate through CV splits
n_splits = 100
cv = TimeSeriesSplit(n_splits=n_splits)

# Create empty array to collect coefficients
coefficients = np.zeros([n_splits, X.shape[1]])

for ii, (tr, tt) in enumerate(cv.split(X, y)):
    # Fit the model on training data and collect the coefficients
    model.fit(X[tr], y[tr])
    coefficients[ii] = model.coef_

######

# Calculate a confidence interval around each coefficient
bootstrapped_interval = bootstrap_interval(coefficients)

# Plot it
fig, ax = plt.subplots()
ax.scatter(feature_names, bootstrapped_interval[0], marker='_', lw=3)
ax.scatter(feature_names, bootstrapped_interval[1], marker='_', lw=3)
ax.set(title='95% confidence interval for model coefficients')
plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()


###Visualizing model score variability over time 
from sklearn.model_selection import cross_val_score

# Generate scores for each split to see how the model performs over time
scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)

# Convert to a Pandas Series object
scores_series = pd.Series(scores, index=times_scores, name='score')

# Bootstrap a rolling confidence interval for the mean score
scores_lo = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=2.5))
scores_hi = scores_series.rolling(20).aggregate(partial(bootstrap_interval, percentiles=97.5))

# Plot the results
fig, ax = plt.subplots()
scores_lo.plot(ax=ax, label="Lower confidence interval")
scores_hi.plot(ax=ax, label="Upper confidence interval")
ax.legend()
plt.show()



###Accounting for non-stationarity
# Pre-initialize window sizes
window_sizes = [25, 50, 75, 100]

# Create an empty DataFrame to collect the stores
all_scores = pd.DataFrame(index=times_scores)

# Generate scores for each split to see how the model performs over time
for window in window_sizes:
    # Create cross-validation object using a limited lookback window
    cv = TimeSeriesSplit(n_splits=100, max_train_size=window)
    
    # Calculate scores across all CV splits and collect them in a DataFrame
    this_scores = cross_val_score(model, X, y, cv=cv, scoring=my_pearsonr)
    all_scores['Length {}'.format(window)] = this_scores

# Visualize the scores
ax = all_scores.rolling(10).mean().plot(cmap=plt.cm.coolwarm)
ax.set(title='Scores for multiple windows', ylabel='Correlation (r)')
plt.show()