'''Plotting multiple graphs 
Strategies
-plotting many graphs on common axes
-creating axes within a figure
-creating subplots within a figure 

import matplotlib.pyplot as plt

plt.plot(t, temperature, 'r') #r for red
#Appears on the same axes
plt.plot(t, dewpoint, 'b') #b for blue
plt.xlabel('Date')
plt.title('Temperature & Dew Point')
#Renders plot objects to screen 
plt.show()

The tool to construct axes explicitly is the axes() command. 

plt.axes([0.05, 0.05, 0.425, 0.9]) #construct the left side of the axes 
plt.plot(t, temperature, 'r')
plt.xlabel('Date')
plt.title('Temperature')
plt.axes([0.525,0.05,0.425,0.9]) #makes new axes on the right of the figure 
plt.plot(t, dewpoint, 'b')
plt.xlabel('Date')
plt.title('Dew Point')
plt.show()

-this displays two graphs side-by-side 

The axes() command
-Syntax axes(): the axes command requires the lower left corner 
suntax: axes( [x_lo, y_lo, width, height] ) # requires the coordinate of the left lower corner 
Units between 0 and 1 (figure dimensions)

The subplot command creates a grid of axes, freeing us from figuring out 

plt.subplot(2, 1, 1)
plt.plot(t, temperature, 'r')
plt.xlabel('Date')
plt.title('Temperature')

plt.subplot(2, 1, 2)
plt.plot(t, dewpoint, 'b')
plt.xlabel('Date')
plt.title('Dew Point')

plt.tight_layout()
plt.show()

The subplot() command
Syntax: subplot(nrows, ncols, nsubplot)
Subplot ordering:
-Row-wise from top left
-Indexed from 1
'''
###Multiple plots 
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()

###Using axes()
# Create plot axes for the first line plot
plt.axes([0.05, 0.05, 0.425, 0.9])

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')

# Create plot axes for the second line plot
plt.axes([0.525, 0.05, 0.425, 0.9])

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')

# Display the plot
plt.show()


###Using subplot()(1)
# Create a figure with 1x2 subplot and make the left subplot active
plt.subplot(1, 2, 1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the right subplot active in the current 1x2 subplot grid
plt.subplot(1, 2, 2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Use plt.tight_layout() to improve the spacing between subplots
plt.tight_layout()
plt.show()

##Using subplot()(2)
# Create a figure with 2x2 subplot layout and make the top left subplot active
plt.subplot(2, 2, 1)

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Make the top right subplot active in the current 2x2 subplot grid 
plt.subplot(2, 2, 2)

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Make the bottom left subplot active in the current 2x2 subplot grid
plt.subplot(2, 2, 3)

# Plot in green the % of degrees awarded to women in Health Professions
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Make the bottom right subplot active in the current 2x2 subplot grid
plt.subplot(2, 2, 4)

# Plot in yellow the % of degrees awarded to women in Education
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()


'''
Zooming in on a specific region of a graph can be achieved with axis(), xlim(), ylim()

Controlling axis extents
axis([xmin, xmax, ymin, ymax])
Control over individual axis extents

xlim([xmin, xmax])
ylim([ymin, ymax])

Can use tuples, list for extents 
e.g. xlim((-2, 3)) works
e.g. xlim([-2, 3]) works also

GDP over time
import matplotlib.pyplot as plt
plt.plot(yr, gdp)
plt.xlabel('Year')
plt.xlabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.show()

Zooming in to a psecific region of the graph generated 
Using xlim()
plt.plot(yr, gdp)
plt.xlabel('Year')
plt.ylabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.xlim((1947, 1957))

plt.show()

Using xlim() and ylim()
plt.plot(yr, gdp)
plt.xlabel('Year')
plt.xlabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.xlim((1947, 1957))
plt.ylim((0, 1000))

plt.show()

Now we can set the horizonatal limits and the vertical limits
plt.plot(yr, gdp)
plt.xlable('Year')
plt.ylabel('Billions of Dollars')
plt.title('US Gross Domestic Product')

plt.axis((1947, 1957, 0, 600))

plt.show()


Other axis() options

Invocation          Result
axis('off')         turns off axis lines, labels
axis('equal')       equal scaling on x and y axes
axis('square')      forces square plot
axis('tight')       sets xlim, ylim to show all data

Using axis('equal')
plt.sublot(2, 1, 1)
plt.plot(x, y, 'red')
plt.title('default axis')
plt.subplot(2, 1, 2)
plt.plot(x, y, 'red')

plt.axis('equal')

plt.title('axis equal')
plt.tight_layout()
plt.show()
    

'''

###Using xlim(), ylim()
# Plot the % of degrees awarded to women in Computer Science and the Physical Sciences
plt.plot(year,computer_science, color='red') 
plt.plot(year, physical_sciences, color='blue')

# Add the axis labels
plt.xlabel('Year')
plt.ylabel('Degrees awarded to women (%)')

# Set the x-axis range
plt.xlim((1990, 2010))

# Set the y-axis range
plt.ylim((0, 50))

# Add a title and display the plot
plt.title('Degrees awarded to women (1990-2010)\nComputer Science (red)\nPhysical Sciences (blue)')
plt.show()

# Save the image as 'xlim_and_ylim.png'
plt.savefig('xlim_and_ylim.png')

###Using axis()
# Plot in blue the % of degrees awarded to women in Computer Science
plt.plot(year,computer_science, color='blue')

# Plot in red the % of degrees awarded to women in the Physical Sciences
plt.plot(year, physical_sciences,color='red')

# Set the x-axis and y-axis limits
plt.axis((1990, 2010, 0, 50))

# Show the figure
plt.show()

# Save the figure as 'axis_limits.png'
plt.savefig('axis_limits.png')


'''
Legends, annotations and styles 

Legends - provide labels for overlaid points and curves 


import matplotlib.pypot as plt 
plt.scatter(setosa_len, setosa_wid, 
            marker='o', color='red', label='setosa')

plt.scatter(setosa_len, setosa_wid, 
            marker='o', color='green', label='versicolor')

plt.scatter(setosa_len, setosa_wid, 
            marker='o', color='blue', label='virginica')

plt.legend(loc='upper right')
plt.title('Iris data')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()

Legend locations 
string          code 
'upper left'    2
'center left'   6
'lower left'    3
'upper center'  9
'center'        10
'lower center'  8

string          code
'upper right'   1
'center right'  7
'lower right'   4
'right'         5
'best'          0

The annotate function adds text to a figure 
Text labels and arrows using annotate() method
There are flexible ways to specify coordinate 
keyword arrowprops:dict of arrow properties 
    - width
    - color
    - etc.

Using annotate() for text
plt.annotate('setosa', xy=(5.0, 3.5))
plt.annotate('virginica', xy=(7.25, 3.5))
plt.annotate('versicolor', xy=(5.0, 2.0))
plt.show()

Options for annotate()
options     description
s           text of label
xy          coordinates to annotate
xytext      coordinates of label
arrowprops  cotrols drawing of arrow


plt.annotate('setosa', xy=(5.0, 3.5),
                xytext=(4.25, 4.0),
                arrowprops={'color':'red'})
plt.annotate('virginica', xy=(7.2, 3.6),
                xytext=(6.5, 4.0),
                arrowprops={'color':'blue})
plt.annotate('versicolor', xy=(5.05, 1.95),
                xytext=(5.5, 1.75),
                arrowprops={'color':'green'})
plt.show()


Working with plot styles 
style sheets in matplotlib
Defaults for lines, points, backgrounds, etc.
Switch styles globally with plt.style.use()
plt.style.available:list of styles

import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.style.use('fivethirtyeight')

'''

###Using legend()
# Specify the label 'Computer Science'
plt.plot(year, computer_science, color='red', label='Computer Science') 

# Specify the label 'Physical Sciences' 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')

# Add a legend at the lower center
plt.legend(loc='lower center')

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()


###Using annotate()
# Compute the maximum enrollment of women in Computer Science: cs_max
cs_max = computer_science.max()

# Calculate the year in which there was maximum enrollment of women in Computer Science: yr_max
yr_max = year[computer_science.argmax()]

# Plot with legend as before
plt.plot(year, computer_science, color='red', label='Computer Science') 
plt.plot(year, physical_sciences, color='blue', label='Physical Sciences')
plt.legend(loc='lower right')

# Add a black arrow annotation
plt.annotate('Maximum', xy=((yr_max, cs_max)), xytext=((yr_max+5, cs_max+5)), arrowprops=dict(facecolor='k'))

# Add axis labels and title
plt.xlabel('Year')
plt.ylabel('Enrollment (%)')
plt.title('Undergraduate enrollment of women')
plt.show()

###
# Import matplotlib.pyplot
import matplotlib.pyplot as plt

# Set the style to 'ggplot'
plt.style.use('ggplot')

# Create a figure with 2x2 subplot layout
plt.subplot(2, 2, 1) 

# Plot the enrollment % of women in the Physical Sciences
plt.plot(year, physical_sciences, color='blue')
plt.title('Physical Sciences')

# Plot the enrollment % of women in Computer Science
plt.subplot(2, 2, 2)
plt.plot(year, computer_science, color='red')
plt.title('Computer Science')

# Add annotation
cs_max = computer_science.max()
yr_max = year[computer_science.argmax()]
plt.annotate('Maximum', xy=(yr_max, cs_max), xytext=(yr_max-1, cs_max-10), arrowprops=dict(facecolor='black'))

# Plot the enrollmment % of women in Health professions
plt.subplot(2, 2, 3)
plt.plot(year, health, color='green')
plt.title('Health Professions')

# Plot the enrollment % of women in Education
plt.subplot(2, 2, 4)
plt.plot(year, education, color='yellow')
plt.title('Education')

# Improve spacing between subplots and display them
plt.tight_layout()
plt.show()

'''
Working with 2D arrays aka raster data - represent either images or 
functions of two variables, also known as bivariate funtions 

Numpy arrays
-Homogeneous in type
-Calculations all at once
They support vectorized computations or calculations over the entire array without 
writing loops 
-Indexing with brackets:
**A[index] for 1D array
**A[index0, index1] for 2D array
-Slicing: 1D arrays: A[slice], 2D arrays: A[slice0, slice1]
**splcing; slice = start:stop:stride
indexes from start to stop-1 in steps of stride

Using meshgrid()
meshgrids.py:

impoort numpy as np
u = p.linspace(-2,2,3)
v = np.linspace(-1,1,5)
X,Y = np.meshgrid(u, v)

import numpy as np
import matplotlib.pyplot as plt
u = np.linspace(-2, 2, 3)
v = np.linspace(-1, 1, 5)
X, Y = np.meshgrid(u, v)

Z = X**2/25 + Y**2/4

print(Z)
plt.set_cmap('gray')
plt.pcolor(Z)
plt.show()
#dark pixels on the graph are closer to 0 than the lighter squares 

Orientations of 2D arrays & images 
orientation.py

import numpy as np
import matplotlib.pyplot as plt
Z = np.array([[1, 2, 3], [4, 5, 6]])
print(z)
plt.pcolor(Z)
plt.show()

When pcolor() plots pixels, values increase from 1 to 6 with values increasing from 
left to right, then vertically from bottom to top starting from the bottom left corner 


'''

###Generating meshes 
# Import numpy and matplotlib.pyplot
import numpy as np
import matplotlib.pyplot as plt

# Generate two 1-D arrays: u, v
u = np.linspace(-2, 2, 41)
v = np.linspace(-1, 1, 21)

# Generate 2-D arrays from u and v: X, Y
X,Y = np.meshgrid(u, v)

# Compute Z based on X and Y
Z = np.sin(3*np.sqrt(X**2 + Y**2)) 

# Display the resulting image with pcolor()
plt.pcolor(Z)
plt.show()

# Save the figure to 'sine_mesh.png'
plt.savefig('sine_mesh.png')


'''
Visualizing bivariate functions 
Pseudocolor pot 

import numpy as np
import matplotlib.pyplot as plt
u = np.linspace(-2, 2, 65)
v = np.linspace(-1, 1, 33)
X, Y = np.meshgrig(u, v)
Z = X**2/25 + Y**2/4
plt.pcolor(Z)
plt.show()

#This is a multi color plot instead of a plot that is grey scale 
#To unnderstand the colors lets display the colors 

plt.pcolor(Z)
plt.colorbar()
plt.show()


plt.pcolor(Z, cmap= 'autumn') #cmap can be 'grey' for grey scale or any other colors
plt.colorbar()
plt.show()

#To fix the color around the psuedo plot use plt.axis('tight')

plt.pcolor(Z)
plt.colorbar()
plt.axis('tight')
plt.show()

# X, Y are 20 meshgrid
plt.pcolor(X,Y,Z) #this executes p color X, Y, Z rather than just Z
plt.colorbar()
plt.show()


#Contour plots can be used when the data varies continuously 
plt.contour(Z)
plt.show()

#The number of contours can be specified 
plt.contour(Z, 30)
plt.show()

#contuour plot using meshgrid
plt.contour(X, Y, Z, 30)
plt.show()

#Filled contour plots 
plt.contourf(X, Y, Z, 30)
plt.colorbar()
plt.show()

More information 
-API has many (optional) keyword arguments 
-More in matplotlib.pyplot documentation
-More examples: http://matplotlb.org/gallery.html
'''

###Contour & filled contour plots
# Generate a default contour map of the array Z
plt.subplot(2,2,1)
plt.contour(X, Y, Z)

# Generate a contour map with 20 contours
plt.subplot(2,2,2)
plt.contour(X, Y, Z, 20)

# Generate a default filled contour map of the array Z
plt.subplot(2,2,3)
plt.contourf(X, Y, Z)

# Generate a default filled contour map with 20 contours
plt.subplot(2,2,4)
plt.contourf(X, Y, Z, 20)

# Improve the spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()


###Modifying colormmaps 

# Create a filled contour plot with a color map of 'viridis'
plt.subplot(2,2,1)
plt.contourf(X,Y,Z,20, cmap='viridis')
plt.colorbar()
plt.title('Viridis')

# Create a filled contour plot with a color map of 'gray'
plt.subplot(2,2,2)
plt.contourf(X,Y,Z,20, cmap='gray')
plt.colorbar()
plt.title('Gray')

# Create a filled contour plot with a color map of 'autumn'
plt.subplot(2,2,3)
plt.contourf(X,Y,Z,20, cmap='autumn')
plt.colorbar()
plt.title('Autumn')

# Create a filled contour plot with a color map of 'winter'
plt.subplot(2,2,4)
plt.contourf(X,Y,Z,20, cmap='winter')
plt.colorbar()
plt.title('Winter')

# Improve the spacing between subplots and display them
plt.tight_layout()
plt.show()


'''
Visualizing bivariate distributions 

Distributions of 2D points 
-2D points given as two 1D arrays x and y
-Goal: generate a 2D histogram from x and y

Histograms in 1D
-Choose bins(intervals)
-Count the realizations within bins 

counts, bins, pathes = plt.hist(x, bins=25)
plt.show()

Bins in 2D
-Different shapes available for binning points 
-Common chouces: rectables & hexagons

In one dimension, straight-line segments are the only possible shape for bins in a histogram 

hist2d(): Rectangular binning 
#X & y are 10 arrays of the same length 
plt.hist2d(x, y, bins=(10, 20)) # bins = (horizontal, vertical)
plt.colorbar()
plt.xlabel('weight ($\\marthrm{kg}')
plt.ylabel('acceleration ($\\mathrm{ms}^{-2}$)')
plt.show()


hexbin(): Gexagonal binning 
plt.hexbin(x, y, gridsize=(15,10)
plt.colorbar()
plt.xlabel('weight ('weight ($\\mathrm{kg}$)')
plt.ylabel('acceleration ($\\mathrm{ms}^{-2}$)}')
plt.show()
'''


###Using hist2d()
# Generate a 2-D histogram
plt.hist2d(hp, mpg, bins=(20, 20),range=((40, 235), (8, 48)))

# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hist2d() plot')
plt.show()


###Using hexbin()
# Generate a 2d histogram with hexagonal bins
plt.hexbin(hp, mpg, gridsize=(15,12), extent=(40,235,8,48))

           
# Add a color bar to the histogram
plt.colorbar()

# Add labels, title, and display the plot
plt.xlabel('Horse power [hp]')
plt.ylabel('Miles per gallon [mpg]')
plt.title('hexbin() plot')
plt.show()


'''
Working with images 
Image is a matrix of intessity value 
-Grayscale images: rectangular 2D arrays
-Color Images: typically three 2D arrays (channels)
--RGB (Red-Green-Blue)
--Channel values:
---0 to 1 (floating-point numbers)
---0 to 255 (8 bit integers)

Loading images 
img = plt.imread('sunflower.jpg')
print(img.shape)

plt.imshow(img)
plt.axis('off')
plt.show()

Reduction to grey-scale image 
collapsed = img.mean(axis=2)
print(collapsed.shape)

Since the image consist of matrices with entries between 0 and 255 using the average 
works as a reasonable proxy for the RGB values when collapsing them to a single scalar intensity  

plt.set_cmap('grey')
plt.imshow(collapsed, cmap='gray')
plt.axis('off')
plt.show()

Uneven saples 
# nonuniform subsampling
uneven = collapsed[::4, ::2]
print(uneven.shape)

plt.imshow(uneven, aspect=2.0)
plt.axis('off')
plt.show()

plt.imshow(uneven, cmap='gray', extent=(0,640,0,480)) #The order of arguments for extent is from left to right and bottom to top
plt.axis('off')
plt.show()
'''

###Loading, examining images
# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Print the shape of the image
print(img.shape)

# Display the image
plt.imshow(img)

# Hide the axes
plt.axis('off')
plt.show()

###Psudocolor plot from image data 
# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Print the shape of the image
print(img.shape)

# Compute the sum of the red, green and blue channels: intensity
intensity = img.sum(axis=2)

# Print the shape of the intensity
print(intensity.shape)

# Display the intensity with a colormap of 'gray'
plt.imshow(intensity, cmap='gray')

# Add a colorbar
plt.colorbar()

# Hide the axes and show the figure
plt.axis('off')
plt.show()

###Extent and aspect
# Load the image into an array: img
img = plt.imread('480px-Astronaut-EVA.jpg')

# Specify the extent and aspect ratio of the top left subplot
plt.subplot(2,2,1)
plt.title('extent=(-1,1,-1,1),\naspect=0.5') 
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=0.5)

# Specify the extent and aspect ratio of the top right subplot
plt.subplot(2,2,2)
plt.title('extent=(-1,1,-1,1),\naspect=1')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=1)

# Specify the extent and aspect ratio of the bottom left subplot
plt.subplot(2,2,3)
plt.title('extent=(-1,1,-1,1),\naspect=2')
plt.xticks([-1,0,1])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-1,1,-1,1), aspect=2)

# Specify the extent and aspect ratio of the bottom right subplot
plt.subplot(2,2,4)
plt.title('extent=(-2,2,-1,1),\naspect=2')
plt.xticks([-2,-1,0,1,2])
plt.yticks([-1,0,1])
plt.imshow(img, extent=(-2,2,-1,1), aspect=2)

# Improve spacing and display the figure
plt.tight_layout()
plt.show()

###Rescaling pixel intensities
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Extract minimum and maximum values from the image: pmin, pmax
pmin, pmax = image.min(), image.max()
print("The smallest & largest pixel intensities are %d & %d." % (pmin, pmax))

# Rescale the pixels: rescaled_image
rescaled_image = 256*(image - pmin) / (pmax - pmin)
print("The rescaled smallest & largest pixel intensities are %.1f & %.1f." % 
      (rescaled_image.min(), rescaled_image.max()))

# Display the rescaled image
plt.title('rescaled image')
plt.axis('off')
plt.imshow(rescaled_image)

plt.show()


'''
Visualizing regression 
import padas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset('tips')
sns.lmplot(x='total_bill',
            y='tip',
            data=tips)
plt.show()

Seaborn and lmplot() make it easy to color observations & regressive curves by factor 
using, in this case, hue='sex'

sns.lmplot(x='total_bill', y='tip',
            data=tips,
            hue='sex',
            palette='Set1')
plt.show()

sns.lmplot(x='total_bill', y='tip',
            data=tips,
            col='sex') #Using col=sex produces this plot of separate factors. 

using residplot()
sns.residplot(x='age', y='fare',
                data=tips,
                color='indianred')
plt.show()

-Similar arguments as implot() but more flexible
-x, y can be arrays or strings 
-data is DataFrame (optional)
-Optional arguments (eg color) as in matplotlib
'''

###Simple linear regression 
# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Plot a linear regression between 'weight' and 'hp'
sns.lmplot(x='weight', y='hp', data=auto)

# Display the plot
plt.show()


###Plotting residuals of a regression 

# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a green residual plot of the regression between 'hp' and 'mpg'
sns.residplot(x='hp', y='mpg', data=auto, color='green')

# Display the plot
plt.show()

'''
A principal difference between sns.lmplot() and sns.regplot() is the way in which 
matplotlib options are passed (sns.regplot() is more permissive).

For both sns.lmplot() and sns.regplot(), the keyword order is used to control the 
order of polynomial regression.

The function sns.regplot() uses the argument scatter=None to prevent plotting the 
scatter plot points again. 
'''

###Higher-order regressions
# Generate a scatter plot of 'weight' and 'mpg' using red circles
plt.scatter(auto['weight'], auto['mpg'], label='data', color='red', marker='o')

# Plot in blue a linear regression of order 1 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, scatter=None, color='blue', label='First Order')

# Plot in green a linear regression of order 2 between 'weight' and 'mpg'
sns.regplot(x='weight', y='mpg', data=auto, scatter=None,color='green', label='Second Order', order=2)

# Add a legend and display the plot
plt.legend(loc='upper right')
plt.show()

###Grouping linear regression by hue 
# Plot a linear regression between 'weight' and 'hp', with a hue of 'origin' and palette of 'Set1'
sns.lmplot(x='weight', y='hp', data=auto, hue='origin', palette='Set1')

# Display the plot
plt.show()

###Grouping linear regressions by row or column 
# Plot linear regressions between 'weight' and 'hp' grouped row-wise by 'origin'
sns.lmplot(x='weight', y='hp', row='origin', data=auto)

# Display the plot
plt.show()


'''
Visualizing univariate distributions 

Visualizing data 
-Univariate -> "one variable" 
-Vialization techniques for sampled univariate data 
*Strip plots 
*Swarm plots 
*Violin plots 

Strip plot draws values on a number line to visualize samples of a single random variabe 

Using stipplot()
sns.striplot(y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()

Grouping into categories 

sns.stripplot(x='day', y='tip', data=tip)
plt.ylabel('tip ($)')
plt.show()

Same values are drawn on top of eachother, to compensate for this we can use jitter 

sns.stripplot(x='day', y='tip', data=tip, size=4, jitter=True)
plt.ylabel('tip ($)')
plt.show() 

swarm plots automatical spread same values on the horizontal axes 

sns.swarmplot(x='day', y='tip', data=tips, hue='sex')
plt.ylabel('tip ($)')
plt.show()

Changing orientation 
sns.swarmplot(x='tip', y='day', data=tips, hue='sex', orient='h')
plt.xlabel('tip ($)')
plt.show()


With lots of data box plot and violin plots are ideal 
Box plots are illustrations of ranges: min, max & med values of a dataset along with the 
third quartile and outliers 

Violin plots show curved distributions (notably a kernal density estimate or KDE that approximates a histogram)
wraapped around a box plot rather than discrete points. The basic idea is that the distribution is denser where 
the violin plot is thicker 

plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')

plt.subplot(1,2,2)
sns.violinplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')

plt.tight_layout()
plt.show()

sns.violinplot(x='day', y='tip', data=tips, inner=None, color='lightgray')
sns.stripplot(x='day', y='tip', data=tips, size=4, jitter=True)
plt.ylabel('tip ($)')
plt.show()

'''

###Constructing stip plots
# Make a strip plot of 'hp' grouped by 'cyl'
plt.subplot(2,1,1)
sns.stripplot(x='cyl', y='hp', data=auto)

# Make the strip plot again using jitter and a smaller point size
plt.subplot(2,1,2)
sns.stripplot(x='cyl', y='hp', data=auto, size=3, jitter=True)

# Display the plot
plt.show()

##Constructing swarm plots 
# Generate a swarm plot of 'hp' grouped horizontally by 'cyl'  
plt.subplot(2,1,1)
sns.swarmplot(x='cyl', y='hp', orient='h', data=auto)

# Generate a swarm plot of 'hp' grouped vertically by 'cyl' with a hue of 'origin'
plt.subplot(2,1,2)
sns.swarmplot(x='hp', y='cyl', orient='h', hue='origin', data=auto)

# Display the plot
plt.show()

###Constructing violin plots
# Generate a violin plot of 'hp' grouped horizontally by 'cyl'
plt.subplot(2,1,1)
sns.violinplot(x='cyl', y='hp', data=auto)

# Generate the same violin plot again with a color of 'lightgray' and without inner annotations
plt.subplot(2,1,2)
sns.violinplot(x='cyl', y='hp', data=auto, inner=None, color='lightgray')

# Overlay a strip plot on the violin plot
sns.stripplot(x='cyl', y='hp', data=auto, size=1.5, jitter=True)

# Display the plot
plt.show()


'''
Visualizing multivariate distributions 
Visualizing data 

-Bivariate -> two variables 
-Multivariate -> multiple variables

Visualizing relationsips in multivariate data 
-Joint plots 
-pair plots 
-Heat maps 


In joint plots the pearson corelation coefficient and p value is displayed in the upper right hand corner 
The pearson correlation coefficient quantifies how strongly two variables are correlated, and the p-valu tells us 
the statistical significance of the difference of this value from 0; see our statistical curriculum for more 
details 

sns.jointplot(x='total_bill', y='tip', data=tips)
plt.show()

Using kde ****You like this one 
sns.jointplot(x='total_bill', y='tip', data=tips, kind='kde')
plt.show()

pair plot

sns.pairplot(tips)
plt.show()

You can stratify the data 
sns.pairplots(tips, hue='sex')
plt.show()

Using heatmap()
print(covariance)
sns.heatmap(covariance)
plt.title('Covariance plot')
plt.show()
'''

###Plotting joiont distributions (1)
# Generate a joint plot of 'hp' and 'mpg'
sns.jointplot(x='hp', y='mpg', data=auto)

# Display the plot
plt.show()

###Plotting joint distributions (2)
# Generate a joint plot of 'hp' and 'mpg' using a hexbin plot
sns.jointplot(x='hp', y='mpg', data=auto, kind='hex')

# Display the plot
plt.show()


###Ploting distributions pairwise (1)
'''
The function sns.pairplot() constructs a grid of all joint plots pairwise 
from all pairs of (non-categorical) columns in a DataFrame. The syntax is very simple: 
sns.pairplot(df), where df is a DataFrame.
'''
# Print the first 5 rows of the DataFrame
print(auto.head())

# Plot the pairwise joint distributions from the DataFrame 
sns.pairplot(auto)

# Display the plot
plt.show()


###Ploting distributions pairwise (2)
# Print the first 5 rows of the DataFrame
print(auto.head())

# Plot the pairwise joint distributions grouped by 'origin' along with regression lines
sns.pairplot(auto, kind='reg', hue='origin')

# Display the plot
plt.show()

###Visualizing correlations with heatmap 
# Print the covariance matrix
print(cov_matrix)

# Visualize the covariance matrix using a heatmap
sns.heatmap(cov_matrix)

# Display the heatmap
plt.show()


###Visualizing time series 
# Print the covariance matrix
print(cov_matrix)

# Visualize the covariance matrix using a heatmap
sns.heatmap(cov_matrix)

# Display the heatmap
plt.show()


'''
Visualizing time series 
Datetimes & time series 

All of the columns can be plotte with the plot command 
plt.pot(weather)
plt.show()

Time series 
-pandas time series: datetime as index
-Datetime:represents periods or time stamps
-Datetime index: specializing slicing 
*weather['2010-07-04']
*weather['2010-03':'2010-04']
*weather['2010-05']
etc.

temperature = weather['Temperature']
march_apr = temperature['2010-03':'2010-04'] #data of March & April 2o1o only 
march_apr.shape

march_apr.iloc[-4:] #extract last 4 entries from time series 

Plotting time series slices 

plt.plot(temperature['2010-01'],
        color='red',
        label='Temperature')
       
dew_point = weather['DewPoint']
plt.plot(dewpoint['2010-01'],
        color='blue',
        label='Dewpoint')

plt.legend(loc='upper right')
plt.xticks(rotation=60)
plt.show()


Selecting & formatting dates 
jan = temperature['2010-01']
dates = jan.index[::96] # Pick every 4th day - this extracts a slice with a stride 96
print(dates)

labels= dates.strftime('%b %d') # Make formatted labels 
print(labels)

Cleaning up the tics on the axis 
plt.plot(temperature['2010-01'], color='red', label='Temperature')
plt.plot(dewpoint['2010-01'], color='blue', label='Dewpoint')

plt.xticks(dates, labels, rotation=60)

plt.legend(loc='upper right')
plt.show()
'''

###Multiple series on common axes 
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot the aapl time series in blue
plt.plot(aapl, color='blue', label='AAPL')

# Plot the ibm time series in green
plt.plot(ibm, color='green', label='IBM')

# Plot the csco time series in red
plt.plot(csco, color='red', label='CSCO')

# Plot the msft time series in magenta
plt.plot(msft, color='magenta', label='MSFT')

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

# Display the plot
plt.show()

###Multiple time series slices (1)
# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Plot the aapl time series in blue
plt.plot(aapl, color='blue', label='AAPL')

# Plot the ibm time series in green
plt.plot(ibm, color='green', label='IBM')

# Plot the csco time series in red
plt.plot(csco, color='red', label='CSCO')

# Plot the msft time series in magenta
plt.plot(msft, color='magenta', label='MSFT')

# Add a legend in the top left corner of the plot
plt.legend(loc='upper left')

# Specify the orientation of the xticks
plt.xticks(rotation=60)

# Display the plot
plt.show()


###Multiple time series slice (2)
# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view_1 = aapl['2007-11':'2008-04']

# Plot the sliced series in the top subplot in red
plt.subplot(2, 1, 1)
plt.xticks(rotation=45)
plt.title('AAPL: Nov. 2007 to Apr. 2008')
plt.plot(view_1, color='red')

# Reassign the series by slicing the month January 2008
view_2 = aapl['2008-01']

# Plot the sliced series in the bottom subplot in green
plt.subplot(2, 1, 2)
plt.xticks(rotation=45)
plt.title('AAPL: Jan. 2008')
plt.plot(view_2, color='green')

# Improve spacing and display the plot
plt.tight_layout()
plt.show()



###Plotting an inset view 
# Slice aapl from Nov. 2007 to Apr. 2008 inclusive: view
view = aapl['2007-11':'2008-04']

# Plot the entire series 
plt.plot(aapl)
plt.xticks(rotation=45)
plt.title('AAPL: 2001-2011')

# Specify the axes
plt.axes([0.25,0.5,0.35,0.35])

# Plot the sliced series in red using the current axes
plt.plot(view, color='red')
plt.xticks(rotation=45)
plt.title('2007/11-2008/04')
plt.show()

'''
Time series with moving windows 
Moving windows and time series 
-Moving window calculations 
-Averages 
-Medians
-Standard deviations 
-Extracts information on longer time scales 

# smoothed computing using moving averages 
smoothed.info()
print(smoothed.iloc[:3,:])

#Viewing 24 hour averages 
#moving average over 24 hours 
plt.plot(smoothed['1d'])
plt.title('Temperature (2010)')
plt.xticks(rotation=60)
plt.show()

#plot Dataframe for January 
plt.plot(smoothed['2010-01'])
plt.legend(smoothed.columns)
plt.title('Temperature (Jan. 2010)')
plt.xticks(rotatation=60)
plt.show()


#Moving standard deviations 
plt.plot(variances['2010-01'])
plt.legend(variances.columns)
plt.title('Temperature deviation')
plt.x
ticks(rotation=60)
plt.show()
'''

###Plotting moving averages 
# Plot the 30-day moving average in the top left subplot in green
plt.subplot(2, 2, 1)
plt.plot(mean_30, 'green')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('30d averages')

# Plot the 75-day moving average in the top right subplot in red
plt.subplot(2, 2, 2)
plt.plot(mean_75, 'red')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('75d averages')

# Plot the 125-day moving average in the bottom left subplot in magenta
plt.subplot(2, 2, 3)
plt.plot(mean_125, 'magenta')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('125d averages')

# Plot the 250-day moving average in the bottom right subplot in cyan
plt.subplot(2, 2, 4)
plt.plot(mean_250, 'cyan')
plt.plot(aapl, 'k-.')
plt.xticks(rotation=60)
plt.title('250d averages')

# Display the plot
plt.show()


###Plotting moving standard deviations
# Plot std_30 in red
plt.plot(std_30, 'red', label='30d')

# Plot std_75 in cyan
plt.plot(std_75, 'cyan', label='75d')

# Plot std_125 in green
plt.plot(std_125, 'green', label='125d')

# Plot std_250 in magenta
plt.plot(std_250, 'magenta', label='250d')

# Add a legend to the upper left
plt.legend(loc='upper left')

# Add a title
plt.title('Moving standard deviations')

# Display the plot
plt.show()


'''
Histogram equalization in images 
Involves spreading out pixel intensities, so subtle contrast are enhanced 

Image histograms - flattening the 2D array into 1D array to use with his()
 orig=plt.imread('low-contrast-moon.jpg')
 pixels = orig.flatten()
 plt.hist(pixels, bins=256, range=(0, 256), normed=True, color='blue', alpha=0.3) #alpha value value controls transparency 
 plt.show()
 
 Rescaling the image 
 minval, maxval = orig.min(), orig.max()
 print(minval, maxval)
 
 rescaled = (255/(maxval-minval)) * (pixels - minval)
 print(rescaled.min(), rescaled.max())
 
 plt.imshow(rescaled)
 plt.axis('off')
 plt.show()
 
 -Original and rescaled histograms 
 plt.hist(orig.flatten(), bins=256, range=(0,255), normed=True, color='blue', alpa=0.2)
 plt.hist(rescaled.flatten(), bin=256, range=(0,255), normed=True, color='green', alpha=0.2)
 plt.legend(['original', 'rescaled'])
 plt.show()
 
 -image histogram & CDF
 plt.hist(pixels, bins=256, range=(0,256), normed=True, color='blue', alpha=0.3)
 
 plt.twinx()
 orig_cdf, bins, patches = plt.hist(pixels, cumulative=True, bins=256, range=(0.256), normed=True, color='red', alpha=0.3)
 plt.title('Image histogram and CDF')
 plt.xlim((0, 255))
 plt.show()
 
 Equalizing intensiy values 
 new_pixels = np.interp(pixels, bins[:-1], orig_cdf*255)
 new = new_pixels.reshape(orig.shape)
 plt.imshow(new)
 plt.axis('off')
 plt.title('Equalized image')
 plt.show()
 
 plt.hist(new_pixels, bins=256, range=(0,256), normed=True, color='blue, alpha=0.3)
 plt.twinx()
 plt.hist(new_pixels, bins=256, range=(0, 256), normed=True, cumulative=True, clor='red', alpha=0.1)
 plt.title('Equalized image histogram and CDF')
 plt.xlim((0.255))
 plt.show()
'''

###Extracting a histogram from a grayscale image
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image, cmap='gray')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
plt.xlim((0,255))
plt.title('Normalized histogram')
plt.hist(pixels, bins=64, range=(0, 256), normed=True, color='red', alpha=0.4)

# Display the plot
plt.show()

###Cumulative Distribution Funcion from an image histogram 
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Display image in top subplot using color map 'gray'
plt.subplot(2,1,1)
plt.imshow(image, cmap='gray')
plt.title('Original image')
plt.axis('off')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Display a histogram of the pixels in the bottom subplot
plt.subplot(2,1,2)
pdf = plt.hist(pixels, bins=64, range=(0,256), normed=False,
               color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()

# Display a cumulative histogram of the pixels
cdf = plt.hist(pixels, bins=64, range=(0,256),
               normed=True, cumulative=True,
               color='blue', alpha=0.4)
               
# Specify x-axis range, hide axes, add title and display plot
plt.xlim((0,256))
plt.grid('off')
plt.title('PDF & CDF (original image)')
plt.show()


###Equalizing the image histogram 
# Load the image into an array: image
image = plt.imread('640px-Unequalized_Hawkes_Bay_NZ.jpg')

# Flatten the image into 1 dimension: pixels
pixels = image.flatten()

# Generate a cumulative histogram
cdf, bins, patches = plt.hist(pixels, bins=256, range=(0,256), normed=True, cumulative=True)
new_pixels = np.interp(pixels, bins[:-1], cdf*255)

# Reshape new_pixels as a 2-D array: new_image
new_image = new_pixels.reshape(image.shape)

# Display the new image with 'gray' color map
plt.subplot(2,1,1)
plt.title('Equalized image')
plt.axis('off')
plt.imshow(new_image, cmap='gray')

# Generate a histogram of the new pixels
plt.subplot(2,1,2)
pdf = plt.hist(new_pixels, bins=64, range=(0,256), normed=False, color='red', alpha=0.4)
plt.grid('off')

# Use plt.twinx() to overlay the CDF in the bottom subplot
plt.twinx()
plt.xlim((0,256))
plt.grid('off')

# Add title
plt.title('PDF & CDF (equalized image)')

# Generate a cumulative histogram of the new pixels
cdf = plt.hist(new_pixels, bins=64, range=(0,256),
               cumulative=True, normed=True,
               color='blue', alpha=0.4)
plt.show()


###Extracting histograms from a color image 
# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Display image in top subplot
plt.subplot(2,1,1)
plt.title('Original image')
plt.axis('off')
plt.imshow(image)

# Extract 2-D arrays of the RGB channels: red, green, blue
red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]

# Flatten the 2-D arrays of the RGB channels into 1-D
red_pixels = red.flatten()
green_pixels = green.flatten()
blue_pixels = blue.flatten()

# Overlay histograms of the pixels of each color in the bottom subplot
plt.subplot(2,1,2)
plt.title('Histograms from color image')
plt.xlim((0,256))
plt.hist(red_pixels, bins=64, normed=True, color='red', alpha=0.2)
plt.hist(green_pixels, bins=64, normed=True, color='green', alpha=0.2)
plt.hist(blue_pixels, bins=64, normed=True, color='blue', alpha=0.2)

# Display the plot
plt.show()

###Extracting bivariate histograms from a color image 
# Load the image into an array: image
image = plt.imread('hs-2004-32-b-small_web.jpg')

# Extract RGB channels and flatten into 1-D array
red, green, blue = image[:,:,0], image[:,:,1], image[:,:,2]
red_pixels = red.flatten()
green_pixels = green.flatten()
blue_pixels = blue.flatten()

# Generate a 2-D histogram of the red and green pixels
plt.subplot(2,2,1)
plt.grid('off') 
plt.xticks(rotation=60)
plt.xlabel('red')
plt.ylabel('green')
plt.hist2d(red_pixels, green_pixels, bins=(32, 32))

# Generate a 2-D histogram of the green and blue pixels
plt.subplot(2,2,2)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('green')
plt.ylabel('blue')
plt.hist2d(green_pixels, blue_pixels, bins=(32, 32))

# Generate a 2-D histogram of the blue and red pixels
plt.subplot(2,2,3)
plt.grid('off')
plt.xticks(rotation=60)
plt.xlabel('blue')
plt.ylabel('red')
plt.hist2d(blue_pixels, red_pixels, bins=(32, 32))

# Display the plot
plt.show()
