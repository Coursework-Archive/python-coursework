'''
Optimal Parameters
#Checking normality of Michelson data

import numpy as np
import matplotlib.pyplot at plt 
mean = np.mean(michelson_speed_of_light)
std = np.std(michelson_speed_of_light)
samples = np.random.normal(mean, std, size=10000)

CDF with bad estimate of st. dev
What is the standard deviation differs by 50%? 
    Then the CDFs no longer match 

If the mean varies by just 0.01%

optimal parameters 
-Parameter values that bring the model in closest agreement with the data 

When the model is wrong the optimal parameters are not meaningfull

Choosing optimal paraters to results in the best agreement between the 
theoretical model distribution and your data 
'''

###How often do we get no-hitters 
# Seed random number generator
np.random.seed(42)

# Compute mean no-hitter time: tau
tau = np.mean(nohitter_times)

# Draw out of an exponential distribution with parameter tau: inter_nohitter_time
inter_nohitter_time = np.random.exponential(tau, 100000)

# Plot the PDF and label axes
_ = plt.hist(inter_nohitter_time,
             bins=50, normed=True, histtype='step')
_ = plt.xlabel('Games between no-hitters')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()


###Do the data follow our story?
# Create an ECDF from real data: x, y
x, y = ecdf(nohitter_times)

# Create a CDF from theoretical samples: x_theor, y_theor
x_theor, y_theor = ecdf(inter_nohitter_time)

# Overlay the plots
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')

# Margins and axis labels
plt.margins(.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Show the plot
plt.show()


###How is the parameter optimal?
# Plot the theoretical CDFs
plt.plot(x_theor, y_theor)
plt.plot(x, y, marker='.', linestyle='none')
plt.margins(0.02)
plt.xlabel('Games between no-hitters')
plt.ylabel('CDF')

# Take samples with half tau: samples_half
samples_half = np.random.exponential(tau/2, 10000)

# Take samples with double tau: samples_double
samples_double = np.random.exponential(tau*2, 10000)

# Generate CDFs from these samples
x_half, y_half = ecdf(samples_half)
x_double, y_double = ecdf(samples_double)

# Plot these CDFs as lines
_ = plt.plot(x_half, y_half)
_ = plt.plot(x_double, y_double)

# Show the plot
plt.show()

'''
Linear regression by at least squares 

Least squares 
-The process of finding the parameters for which the sum of the squares 
of the residals is minimal

You can use np.polyfit() with linear functions 

slope, intercept = np.polyfit(x, y, polynomial degree)
'''

# Plot the illiteracy rate versus fertility
_ = plt.plot(illiteracy, fertility, marker='.', linestyle='none')
plt.margins(0.02)
_ = plt.xlabel('percent illiterate')
_ = plt.ylabel('fertility')

# Perform a linear regression using np.polyfit(): a, b
a, b = np.polyfit(illiteracy, fertility, 1)

# Print the results to the screen
print('slope =', a, 'children per woman / percent illiterate')
print('intercept =', b, 'children per woman')

# Make theoretical line to plot
x = np.array([0,100])
y = a * x + b

# Add regression line to your plot
_ = plt.plot(x, y)

# Draw the plot
plt.show()


###How is it optimal?
# Specify slopes to consider: a_vals
a_vals = np.linspace(0,0.1,200)

# Initialize sum of square of residuals: rss
rss = np.empty_like(a_vals)

# Compute sum of square of residuals for each value of a_vals
for i, a in enumerate(a_vals):
    rss[i] = np.sum((fertility - a * illiteracy - b)**2)

# Plot the RSS
plt.plot(a_vals, rss, '-')
plt.xlabel('slope (children per woman / percent illiterate)')
plt.ylabel('sum of square of residuals')

plt.show()

'''
Anscombe's quartet 
All of the data sets parameter:
Avg X-values the same 
Avg Y-values the same 
linear regression the same 
sum of the square the residuals
'''

### Linear regression on appropriate anscombe data
# Perform linear regression: a, b
a, b = np.polyfit(x, y, 1)

# Print the slope and intercept
print(a, b)

# Generate theoretical x and y data: x_theor, y_theor
x_theor = np.array([3, 15])
y_theor = x_theor * a + b

# Plot the Anscombe data and theoretical line
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.plot(x_theor, y_theor)

# Label the axes
plt.xlabel('x')
plt.ylabel('y')

# Show the plot
plt.show()


###Linear regression on all Anscombe data 
# Iterate through x,y pairs
for x, y in zip(anscombe_x, anscombe_y):
    # Compute the slope and intercept: a, b
    a, b = np.polyfit(x, y, 1)

    # Print the result
    print('slope:', a, 'intercept:', b)


'''
Generating bootstrap replicates 
Resampling engine: np.random.choice()

import numpy as np
np.random.choice([1, 2, 3, 4, 5], , size=5)

Computing a bootstrap replicate 
bs_sample = np.random.choice(michelson_speed_of_light, size=100)
np.mean(bs_sample)
np.median(bs_sample)
np.std(bs_sample)
'''

###Visualizing Bootstrap Samples
for i in range(50):
    # Generate bootstrap sample: bs_sample
    bs_sample = np.random.choice(rainfall, size=len(rainfall))

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 color='gray', alpha=0.1)

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)
_ = plt.plot(x, y, marker='.')

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')

# Show the plot
plt.show()


'''
Bootstrap replicate function
def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 10 date."""
    bs_sample = np.random.choice(date, len(date))
    return func(bs_sample)
    
bootstrap_replcate_1d(michelson_speed_of_light, np.mean)

bootstrap_replcate_1d(michelson_speed_of_light, np.mean)

bootstrap_replcate_1d(michelson_speed_of_light, np.mean)


bs_replicates = np.emppty(10000)
for i in range(10000):
    bs_replicates[i] = bootstrap_replicate_1d(michelson_speed_of_light)

plotting a histogram of bootstrap replicates 
_ = plt.hist(bs_replicates, bins=30, normed=True) #True set the bars of the histogram so that the total area of the bars is equal to one
_ = plt.xlabel('mean speed of light (km/s)')
_ = plt.ylabel('PDF')
plt.show()

The confidkence interval of a statistic 
-If we repeated measurements over and over again, 
p% of the observed values would lie within the p% confidence interval

conf_int = np.percentile(bs_replicates, [2.5, 97.5])
'''

###Generating many bootstrap replicates 
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates


