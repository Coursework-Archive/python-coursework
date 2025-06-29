''' Exploratory data analysis (EDA)
The process of organizing, plotting, and summarizing a data set 
"Exploratory data analysis can ever be the whole story, but nothing else can serve 
as the foundation stone." - John Tukey 

Ex. (data from - http://www.data.gov/)
import pandas as pd
df_swing = pd.read_csv('2008_swing_state.csv')
df_swing[['state', 'country', 'dem_share']]


import matplotlb.pyplot as plt
_ = plt.hist(df_swing['dem_share']) #can accept dataframe column and numpy array 
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
plt.show() #default to 10 bins

plt.hist returns three arrays and only the plot is wanted, "underscore" to them, a common 
practice in Python

bin_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
_ = plt.hist(df_swing['dem_share'], bins=bin_edges)
plt.show()

Alternatively bins can be separated automatically by using the bins argument with a float

_ = plt.hist(df_swing['dem_share'], bins=20)
plt.show()

Seaborn
An excellent Matplotlib-based statistical data visualization package written by
Michael Waskom

import seaborn as sns
sns.set()
_ = plt.hist(df_swing['dem_share'])
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('number of counties')
plt.show()
'''

###Plotting a histogram of iris data 
##Data collected by Edward Anderson 
##Made famous by Ronald Fisher
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Set default Seaborn style
sns.set()

# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)

# Show histogram
plt.show()

###Axis labels!
# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()

'''
The square root rule when choosing number of bins 
square the number of samples and choose that for the number of bins assigned 

'''
###Adjusting the number of bins in a histogram 
# Import numpy
import numpy as np

# Compute number of data points: n_data
n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot the histogram
plt.hist(versicolor_petal_length, n_bins)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()

'''
Binning bias
The same data may be interpreted differently depending on choiice of bins 
The data is sweeped into bins and the data loses the actual value. To remedy these problems
we can make a bee swarm plot, also called a swarm plot, position along the y-axis is the 
quantitative information. The data along the x axis is spread to make them visible, but there 
precise location is not important. Notably we no longer have binning bias.

Generating a bee swarm plot 
_ = sns.swarmplot(x='state', y='dem_share', data=df_swing)
_ = plt.xlabel('state')
_ = plt.ylabel('percent of vote for Obama')
plt.show()
'''

###Bee swarm plot
# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x='species', y='petal length (cm)', data=df)

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')

# Show the plot
plt.show()

'''
With large datasets if a bee swarm plot is used for visual representation then the edges 
are constructed with overlapping data points, this is obfuscating data. As an alternative 
we can compute an empirical cumulative distribution function (ECDF). 
X-value of an ECDF is the quantity you are measuring 
Y-value is the fraction of data points that have a value smaller than the corresponding 
x-value

import numpy as np
x = np.sort(df_swing['dem_share'])
y = np.arrange(1, len(x)+1) / len(x))
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('percent of vote for Obama')
_ = plt.ylabel('ECDF')
plt.margins(0.02) # Keeps data off plot edges 
plt.show()
'''

###Computing the ECDF
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

###Plotting the ECDF
# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')

# Label the axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()


###Comparison of ECDFs
# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)

# Plot all ECDFs on the same plot
_ = plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
_ = plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()

'''
Quantitative exploratory data
Mean vote percentage 
import numpy as np
np.mean(dem_share_PA)

Median - the middle value of a data set, 50th percentile
np.median(dem_share_UT)
'''

###Computing means 
# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)

# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')

'''
Percentiles, outliers, and box plots
The 25th percentile is the vaue of the data point greater than 25% of the 
sorted data
np.percentile(df_swing['dem_share'],[25, 50, 75]) #this computes the 25th, 50th and 75th 

Box plot 
The total height of the box contains the middle of 50% of the data, this is the interquartile 
range. The whiskers extend a distance of 1.5 x the IQR OR the extent of the data whichever is 
is more extreme. Points ouside of the whiskers are plotted as individual points, which we 
demarcate as outliers

import matplotlib.pyplot as plt
import seaborn as sns
_ = sns.boxplot(x='east_west', y='dem_share', data=df_all_states)
_ = plt.xlabel('region')
_ = plt.xlabel('percent of vote for Obama')
plt.show()
'''

###Computing percentiles 
# Specify array of percentiles: percentiles
percentiles = np.array([2.5, 25, 50, 75, 97.5])


# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

# Print the result
print(ptiles_vers)

###Comparing percentile to ECDF
# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

# Show the plot
plt.show()


###Box-and-whisker plot
# Create box plot with Seaborn's default settings
_ = sns.boxplot(x='species', y='petal length (cm)', data=df)

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')

# Show the plot
plt.show()


'''
Variance 
The mean squared distance of the data from their mean 
Informally, a measure of the spread of data 

np.var(dem_share_FL)

square-root of the variance is the standard deviation
np.std(dem_share_FL)

np.sqrt(np.var(dem_share_FL))
'''

###Computing variance 
# Array of differences to mean: differences
differences = np.array(versicolor_petal_length - np.mean(versicolor_petal_length))

# Square the differences: diff_sq
diff_sq = differences**2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Print the results
print(variance_explicit, variance_np)

###The standard deviation and the variance 
# Compute the variance: variance
variance = np.var(versicolor_petal_length)

# Print the square root of the variance
print(np.sqrt(variance))

# Print the standard deviation
print(np.std(versicolor_petal_length))

'''
Covariance and the Pearson correlation coefficient
Geerating a scatter plot 
_ = plt.plot(total_votes/1000, dem_share, marker='.', linestyle='none')
_ = plt.xlabel('total votes (thousands)')
_ = plt.ylabel('percent of vote for Obama')

Covariance 
A measure of how two quantities vary together 
covariance = 1/n sum(xi - mean(x))(yi - mean(y))

If we want to have a more generally applicable measure of how two varibles depend on 
each other, we want to be dimensionless, that is to not have any units 
Pearson correlation coefficient 
p = Pearson correlation = covariance /(std of x)(std of y) = variability due to codependence / independeant variability
Ranges from -1 for complete anti-correlation to 1 for copmlete correlation

'''

###Scatter plot
# Make a scatter plot
_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')


# Label the axes
_ = plt.xlabel('Versicolor Petal Length (cm)')
_ = plt.ylabel('Versicolor Petal Width (cm)')

# Show the result
plt.show()


###Computing the covariance 
# Compute the covariance matrix: covariance_matrix
covariance_matrix = np.cov(versicolor_petal_length, versicolor_petal_width)

# Print covariance matrix
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1]

# Print the length/width covariance
print(petal_cov)


###Computing the Pearson correlation coefficient 
'''
Pearson correlation coefficient, also caled the Pearson r, is often easier to interpret than the covariance 
It is computed using np.corrcoef(), it returns a 2D array much like np.cov()
'''
def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x, y)


    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length, versicolor_petal_width)

# Print the result
print(r)

'''
Probabilistic logic and statistical inference
Probabilistic reasoning allows us to describe uncertainty 
The process from going to measured data to probalistic conclusions 

Defining the probability that we get heads or tails 
< 0.5 -> heads 
>= 0.5 tails 

Bernoulli trial - an experiment that has to options, "success" (True) and "failure" (False)

Random number seed 
-integer fed into random number generation algorithm 
-Manually seed random number generator if you need reproducibility 
Specified using np.random.seed()

Simulating 4 coin flips 
import numpy as np
np.random.seed(42)
random_numbers = np.random.random(size=4)
random_numbers
heads = random_numbers < 0.5
heads
np.sum(heads)

simulating 4 coin flips 
n_all_heads = 0 # Initialize number of 4-heads trials 
for _ in range(10000):
    heads = np.random.random(size=4) < 0.5
    n_heads = np.sum(heads)
    if n_heads == 4:
        n_all_heads += 1
        
n_all_heads / 10000 #Probabililty of getting all 4 head the number of times we got all heads 
divided by the total number of trials we did

Hacker stats probabilities 
*Determine how to simulate data 
*Simulate many many times 
*Probability is approximately fraction of trials with the outcome of interest 
'''

###Generating random nmbers usinig the np.random module
# Seed the random number generator
np.random.seed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()


###The np.random module and Bernoulli trials 
def perform_bernoulli_trials(n, p):
    """Perform n Bernoulli trials with success probability p
    and return number of successes."""
    # Initialize number of successes: n_success
    n_success = 0

    # Perform trials
    for i in range(n):
        # Choose random number between zero and one: random_number
        random_number = np.random.random()

        # If less than p, it's a success so add one to n_success
        if random_number < p:
            n_success += 1

    return n_success

###How many defaults might we expect?
# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100, 0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()


###Will the bank fail 
# Compute ECDF: x, y
x, y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money
n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))


'''
Probability distributions and stories: The Binomial distribution 

Probabilities mass function (PMF)
-The set of probabilities of discrete outcomes
The outcomes are discrete because only a certain value may be attained; you cannot 
roll a 3-point-7 with a die
 
Binomial distribution: the story 
-The number r of successes in n Bernoulli trials with probability p of success, is Binomially 
distributed 
-The number r of heads in 4 coin flips with probability 0.5 of heads, is Binomially distributed 

Sampling from the Binomial distribution 
np.random.binomial(4, 0.5)
array([4, 3, 2, 1, 1, 0, 3, 2, 3, 0])
samples = np.random.binomial(60, 0.1, size=10000)
n = 60
p = 0.1

The Binomial CDF
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
x, y = ecdf(samples)
_ = plt.plot(x, y, marker='.', linestye='none')
plt.margins(0.02)
_ = plt.xlabel('number of successes')
_ = plt.ylabel('CDF')
plt.show()
'''

###Sampling out of the Binomia distribution
# Take 10,000 samples out of the binomial distribution: n_defaults
n_defaults = np.random.binomial(n=100, p=0.05, size=10000)

# Compute CDF: x, y
x, y = ecdf(n_defaults)

# Plot the CDF with axis labels
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('number of defaults out of 100')
_ = plt.ylabel('CDF')

# Show the plot
plt.show()

###Plotting the Binomial PMF
# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
_ = plt.hist(n_defaults, normed=True, bins=bins)

# Label axes
_ = plt.xlabel('number of defaults out of 100')
_ = plt.ylabel('CDF')


# Show the plot
plt.show()

'''
Poisson processes and the Poison distribution 
The timing of the next event is completely indeendent of when the previous event happened 
Exampes of Poisson processes 
*Natural births in a given hospital 
*Hit on a website during a given hour 
*Meteor strikes 
*Molecular collision in a gas
*Aviation incidents 
*Buses in Poissonville

Poisson distribution 
*The number r of arrivals of a Poisson process in a given time interval with average rate of ? 
arrivals per interval is Poisson distributed 
*The number r of hits on a website in one hour with an average hit rate of 6 hits per hour 
is Poisson distributed 

Poisson Distribution
*Limit of the Binomial distribution for low probability of success and large number of trials 
*That is, for rare events 

The Poisson CDF samples = np.random.poisson(6, size=10000)
x, y = ecdf(samples)
_ = plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.2)
_ = plt.xlabel('number of successes')
_ = plt.ylabel('CDF')
plt.show()


'''
###Relationship between Binomial and Poisson distributions 

# Draw 10,000 samples out of Poisson distribution: samples_poisson
samples_poisson = np.random.poisson(10, size=10000)

# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p
n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(n[i], p[i], size=10000)

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))


'''
You can use the Poisson distribution. But remember: the Poisson 
distribution is a limit of the Binomial distribution when the probability 
of success is small and the number of Bernoulli trials is large.
'''

###Was 2015 anomalous?
# Draw 10,000 samples out of Poisson distribution: n_nohitters
n_nohitters = np.random.poisson(251/115, size=10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large / 10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)

'''
Probabiliy density functions
Continuous variables- quantities that can take any vaue, not just discrete vaues 

Michelson's speed of light experiment, 1880 - megameters (1000 km/s)

Probability density function (PDF)
*Continuous analog to the PMF
*Mathematical description of the relative likelihood of observing a value of a 
continuous variable

Areas under the pdf give probabilities 
to do this calculation we are really just looking at the cumulative distribution
function (CDF) of the normal distribution  

In Michelson's example the CDF gives the probability the measured spead of light will
be less than the value on the x-axis
'''

'''
Introduction to the Normal distribution 
Describes a continuous variable whose PDF has a single symmetric peak
The normal distribution is parametrized by two parameters, the mean determines where the 
center of the peak is. The standard deviation is a measure of how wide the peak is, 
or how spread out the data are.

Parameter                               Calculated from data 
mean of a               !=              mean computed from 
normal distribution                     data

st. dev. of a           !=              standard deviation
Normal distribution                     computed from data 

Checking Normality of Michelson data 
import numpy as np
mean = np.mean(micheson_speed_of_light)
std = np.std(micheson_speed_of_light)
samples = np.random.normal(mean, std, size=10000)
x, y = ecdf(michelson_speed_of_light)
x_theor, y_theor = ecdf(samples)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', inestyle='none')
_ = plt.xlabel('speed of light (km/s)'))
_ = plt.ylabel('CDF')
plt.show()
'''

###The normal PDF
# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20, 1, size=100000)
samples_std3 = np.random.normal(20, 3, size=100000)
samples_std10 = np.random.normal(20, 10, size=100000)

# Make histograms
_ = plt.hist(samples_std1, bins=100, normed=True, histtype='step')
_ = plt.hist(samples_std3, bins=100, normed=True, histtype='step')
_ = plt.hist(samples_std10, bins=100, normed=True, histtype='step')

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()

###The normal CDF
# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)

# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker='.', linestyle='none')
_ = plt.plot(x_std3, y_std3, marker='.', linestyle='none')
_ = plt.plot(x_std10, y_std10, marker='.', linestyle='none')


# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


'''
The Gaussian distribution also called the normal distribution 
Describes most symmetric peaked data 
'''

###Are the Belmont Stakes results Normally distributed 
# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)


# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

# Get the CDF of the samples and of the data
x, y = ecdf(belmont_no_outliers)
x_theor, y_theor = ecdf(samples)


# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()


###What are the chances of a horse matching or beating Secretariat's record 
# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, size=1000000)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples <= 144) / len(samples)

# Print the result
print('Probability of besting Secretariat:', prob)


'''
The Exponential distribution
The waiting time between arrivals of a Poisson process is Exponentially distributed 

Possibe Poisson process 
*Nuclear incidents:
-Timing of one is independent of all others so the time between incidents should 
be Exponentially distributed

mean = np.mean(inter_times)
samples = np.random.exponential(mean, size=10000)
x, y = ecdf(inter_times)
e_theor, y_theor = ecdf(samples)
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('time (days)')
_ = plt.ylabel('CDF')
plt.show()

The Exponential and Normal are just two of many examples of continuous distributions 
'''

###If you have a story, you can simulate it!
def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)

    return t1 + t2

###Distribution of no-hitters and cycles 
# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764, 715, size=100000)

# Make the histogram
_ = plt.hist(waiting_times, bins=100, normed=True, histtype='step')


# Label axes
_ = plt.xlabel('no-hitter cycle')
_ = plt.ylabel('hitting cycle')


# Show the plot
plt.show()
