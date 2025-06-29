'''
What is Seaborn? 
* Python data visualization library 
* Easily create the most common types of plots 

Advantages of Seaborn 
* Easy to use
* Works well with pandas data strutures 
* Built on top of matplotlib

import seaborn as sns
import matplotlib.pyplot as plt 
height = [62, 64, 69, 75, 66, 68, 65, 71, 76, 73]
weight = [120, 136, 148, 175, 137, 165, 154, 172, 200, 187]
sns.scatterplot(x=height, y=weight)
plt.show()

gender = ["Female", "Female", "Female", "Female", "Male", 
        "Male", "Male", "Male", "Male"]
sns.countplot(x=gender) #data=categorical list 
plt.show()
'''

###Making a scatter plot with lists
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plot with GDP on the x-axis and number of phones on the y-axis
sns.scatterplot(x=gdp, y=phones)

# Show plot
plt.show()


###Making a scatter plot with a list
# Import Matplotlib and Seaborn
import seaborn as sns
import matplotlib.pyplot as plt


# Create count plot with region on the y-axis
sns.countplot(y=region)

# Show plot
plt.show()


'''
Using pandas with seaborn 
* Python library for data analysis
* Easily read datasets from csv, txt, and other types of files 
* Datasets take the form of DataFrame objects 

improt pandas as pd 
df = pd.read_csv("masculinity.csv")
df.head()

Using DataFrames with countplot()

import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
df = pd.read_csv("masculinity.csv")
sns.countplot(x="how_masculine", data=df) #x is thee named column in the dataframe 
plt.show()
'''

###"Tidy" vs. "untidy" data
# Import Matplotlib, Pandas, and Seaborn
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Create a DataFrame from csv file
df = pd.read_csv(csv_filepath)

# Create a count plot with "Spiders" on the x-axis
sns.countplot(data=df, x="Spiders")

# Display the plot
plt.show()

'''
Adding a third variable with hue
import pandas as pd 
import seaborn as sns
tips = sns.load_dataset("tips")
tips.head()

sns.scatterplot(x="total_bill", 
                y="tip",
                data=tips,
                hue="smoker",
                hue_order=["Yes", "No"])
 plt.show()
 
 
 Specifying hue colors 
 import matplotlib.pyplot as plt 
 import seaborn as sns
 hue_colors = {"Yes": "black",
                "No": "red"}
 sns.scatterplot(x="total_bill"
                   y="tip",
                   data=tips,
                   hue="smoker",
                   palette=huecolors)
plt.show()


Using HTML hex color codes with hue 
import matplotlib.pyplot as plt 
import seaborn as sns

hue_colors = {"Yes": "#808080",
                "No": "#00FF00"}

sns.scatterplot(x="total_bill", 
                y="tip",
                data=tips,
                hue="smoker"
                palatte=hue_colors)
plt.show()


Using hue with count plots 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.countplot(x="smoker",
                data=tips,
                hue="sex")

plt.show()
'''


###Hue and scatter plots 
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location")

# Show plot
plt.show()


###Hue and scatter plots
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Change the legend order in the scatter plot
sns.scatterplot(x="absences", y="G3", 
                data=student_data, 
                hue="location", hue_order=["Rural","Urban"])


# Show plot
plt.show()


###Hue and count plots 
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a dictionary mapping subgroup values to colors
palette_colors = {"Rural": "green", "Urban": "blue"}

# Create a count plot of school with location subgroups
sns.countplot(x="school", data=student_data, hue="location", palette=palette_colors)



# Display plot
plt.show()


'''
Introduction to realtional plots and subplots 
relational plot 
Introducing replot()
* Create "relational plots": scatter plots or line plots
Why use replot() instead of scatterplot()?
* replot() lets you create subplots in a single figure 

scatterplot() vs. replot()
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()

Using replot()
import seaborn as sns
import matplotlib.pyplot as plt
sns.replot(x="total_bill", y="tip", data=tips, kind="scatter")
plt.show()

Subplots in columns 
import seaborn as sns
import matplotlib.pyplot as plt
sns.replot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            col="smoker")
plt.show()
            
Subplots in rows 
import seaborn as sns
import matplotlib.pyplot as plt 
sns.replot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            row="smoker")
plt.show()


Wrapping columns
import seaborn as sns
import matplotlib.pyplot as plt
sns.replot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            col="day"
            col_wrap=2)
plt.show()

Ordering columns
import seaborn as sns
import matplotlib.pyplot as plt

sns.replot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            col="day",
            col_wrap=2,
            col_order=["Thur","Fri","Sat","Sun"])
plt.show()
'''

###Creating subplots with col and row
# Change to use relplot() instead of scatterplot()
sns.relplot(x="absences", y="G3", 
                data=student_data)

# Show plot
plt.show()


# Change this scatter plot to arrange the plots in rows instead of columns
sns.relplot(x="absences", y="G3", 
            data=student_data,
            kind="scatter", 
            col="study_time", row="study_time")

# Show plot
plt.show()

###Creating two-factor subplots
# Adjust to add subplots based on school support
sns.relplot(x="G1", y="G3", 
            data=student_data,
            kind="scatter", col="schoolsup",col_order=["yes","no"])

# Show plot
plt.show()

# Adjust further to add subplots based on family support
sns.relplot(x="G1", y="G3", 
            data=student_data,
            kind="scatter", 
            col="schoolsup",
            col_order=["yes", "no"], row="famsup", row_order=["yes","no"])

# Show plot
plt.show()


'''
Customizing scatter plots
Scatter plot overview 
Show relationship between two quantitative variables 

We've seen:
* Subplots(col and row)
* Subgroups with color (hue)
New Customizations:
* Subgroups with point size and style
* Changing point trnsparency
Use with both scatterplot() and replot()

Subgroups with point size 
import seaborn as sns
import matplotlib.pyplot as plt

sns.replot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            size="size",
            hue="size")
 plt.show()
 
 Subgroups with point style 
 import seaborn as sns
 import matplotlib.pyplot as plt
 
 sns.relplot(x="total_bill",
                y="tip",
                data=tips,
                kind="scatter",
                hue="smoker",
                style="smoker")
 
 Changing point transparency
 import seaborn as sns
 import matplotlib.pyplot as plt 
 
 #Set alpha to be between 0 and 1 
 sns.relplot(x="total_bill",
            y="tip",
            data=tips,
            kind="scatter",
            alpha=0.4)
 '''
 
 ###Changing the size of scatter plot points 
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create scatter plot of horsepower vs. mpg
sns.relplot(x="horsepower", y="mpg", 
            data=mpg, kind="scatter", 
            size="cylinders", hue="cylinders")

# Show plot
plt.show()



###Changing the style of scatter plot points
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot of acceleration vs. mpg
sns.relplot(data=mpg, x="acceleration", y="mpg", style="origin", hue="origin", kind="scatter")


# Show plot
plt.show()


'''
Introduction to line plots
What are line plots?
Two types of relational plots: scatter plots and line plots

Scatter plots
* Each plot point is an independent observation 

Line plots 
* Each plot point represents the same "thing", typically tracked over time

Scatter plot 
import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour", y="No_2_mean", data=air_df_mean, kind="scatter")
plt.show()

Line plot 
import matplotlib.pyplot as plt
import seaborrn as sns

sns.relplot(x="hour", y="NO_2_mean", data=air_df_mean, kind="line")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour", y="NO_2_mean", 
            data=air_df_loc_mean, 
            kind="line", 
            style="location", 
            hue="location",
            dashes="False")
plt.show()

import matplotlib.pyplot as plt 
import seaborn as sns

sns.relplot(x="hour", y="NO_2", data=air_df, kind="scatter")
plt.show()

Line plot
import matplotlib.pyplot as plt
import seaborn as sns

sns.relplot(x="hour", y="NO_2", data=air_df, kind="line")
plt.show()

Multiple observations per x-values
Shaped region is the confidence interval
* Assumes dataset is a random sample 
* 95% confident that the mean is within this interval
* Indicates uncertainty in our estimate 

Replacing confidence interval with standard deviation 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.relplot(x="hour", y="NO_2", data=air_df, kind="line", ci="sd")
plt.show()
'''

###Interpreting line plot
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create line plot
sns.relplot(data=mpg, kind="line", x="model_year", y="mpg")


# Show plot
plt.show()

###Visualizing standard deviation with line plots
# Make the shaded area show the standard deviation
sns.relplot(x="model_year", y="mpg",
            data=mpg, kind="line", ci="sd")

# Show plot
plt.show()

###Plotting subgroups in line plots
# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Create line plot of model year vs. horsepower
sns.relplot(kind="line", data=mpg, x="model_year", y="horsepower", ci=None)

# Show plot
plt.show()


# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Change to create subgroups for country of origin
sns.relplot(x="model_year", y="horsepower", 
            data=mpg, kind="line", 
            ci=None, style="origin")

# Show plot
plt.show()

# Import Matplotlib and Seaborn
import matplotlib.pyplot as plt
import seaborn as sns

# Add markers and make each line have the same style
sns.relplot(x="model_year", y="horsepower", 
            data=mpg, kind="line", 
            ci=None, style="origin", 
            hue="origin", markers=True, dashes=False)
# Show plot
plt.show()


'''
Count plots and bar plots
Examples: count plot, bar plot
Involve a categorical variable 
Comparisons between groups 

catplot()
* Used to create categorical plot
* Same advantages of replot()
* Easily create subplots with col= and row=

countplot() vs. catplot()
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="how_masculine", data=masculinity_data)
plt.show()

Changing the order 
import matplotlib.pyplot as plt
import seaborn as sns
category_order = ["No answer", "Not at all", "Not very", "Somewhat", "Very"]

sns.catplot(x="how_masculine", data=masculinity_data, kind="count", order=category_order)
plt.show()

Bar plots
Displays mean of quantitative variable per category 

import matplotlib.pyplot as plt
import seaborn as sns

sns.catplot(x="day", y"total_bill", data=tips, kind="bar")
plt.show()

* Lines show 95% confidence intervals for the mean 
* Shows uncertainty about our estimate 
* Assumes our data is a random sample 

Turning off confidence intervals 
import matplotlib.pyplot as plt 
import seaborn as sns

sns.catplot(x="day", 
            y="total_bill",
            data=tips,
            kind="bar",
            ci=None)
plt.show()


Changing the orientation 
import matplotlib.pyplot as plt
import seaborn as sns

sns.catplot(x="total_bill",
            y="day",
            data=tips,
            kind="bar")
plt.show()
'''

###Count plots
# Create count plot of internet usage
sns.catplot(kind="count",data=survey_data, x="Internet usage")


# Show plot
plt.show()

# Create column subplots based on age category
sns.catplot(y="Internet usage", data=survey_data,
            kind="count")

# Show plot
plt.show()

###Count plots
# Create column subplots based on age category
sns.catplot(y="Internet usage", data=survey_data,
            kind="count", col="Age Category")

# Show plot
plt.show()


###Bar plots with percentages 
# Create a bar plot of interest in math, separated by gender
sns.catplot(data=survey_data, kind="bar", x="Gender", y="Interested in Math")


# Show plot
plt.show()

###Customizing bar plots 
# Rearrange the categories
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="bar")

# Show plot
plt.show()




###Customizing bar plots
# Rearrange the categories
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="bar", order=["<2 hours", "2 to 5 hours", "5 to 10 hours", ">10 hours"])

# Show plot
plt.show()

# Turn off the confidence intervals
sns.catplot(x="study_time", y="G3",
            data=student_data,
            kind="bar",
            order=["<2 hours", 
                   "2 to 5 hours", 
                   "5 to 10 hours", 
                   ">10 hours"], ci=None)

# Show plot
plt.show()


'''
What is a box plot?
* Shows the distribution of quantitative data 
* See median, spread, skewness, and outliers 
* Facilitates comparisons between groups 

How to create a box
import matplotlib.pyplot as plt 
import seaborn as sns

g = sns.catplot(x="time",
                y="total_bill",
                data=tips,
                kind="box")
plt.show()

Omitting the outliers using 'sym'
import matplotlib.pyplot as plt 
import seaborn as sns 

g = sns.catplot(x="time",
                y="total_bill",
                data=tips,
                kind="box",
                sym="")
plt.show()

Changing the whiskers using 'whis'
* By default, the whiskers extend to 1.5 * the interquartile range 
* Make them extend to 2.0 * IQR: whis=2.0
* Show the 5th and 95th percentile: wist=[5, 95]
* Show min and max values: whis=[0, 100]

Changing the whiskers using 'whis'
import matplotlib.pyplot as plt
import seaborn as sns 

g = sns.catplot(x="time",
                y="total_bill",
                data=tips,
                kind="box",
                whis=[0, 100])
plt.show()
'''

###Create and interpret a box plot 
# Specify the category ordering
study_time_order = ["<2 hours", "2 to 5 hours", 
                    "5 to 10 hours", ">10 hours"]

# Create a box plot and set the order of the categories
sns.catplot(data=student_data, x="study_time", y="G3", order=study_time_order, kind="box")


# Show plot
plt.show()

###Omitting outliers 
# Create a box plot with subgroups and omit the outliers
sns.catplot(data=student_data, x='internet', y='G3', kind="box", hue="location", sym='')


# Show plot
plt.show()


###Adjusting the whiskers 
# Extend the whiskers to the 5th and 95th percentile
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=0.5)

# Show plot
plt.show()


###Adjusting the whiskers 
# Extend the whiskers to the 5th and 95th percentile
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=[5,95])

# Show plot
plt.show()


# Set the whiskers at the min and max values
sns.catplot(x="romantic", y="G3",
            data=student_data,
            kind="box",
            whis=[0, 100])

# Show plot
plt.show()


'''
Point plots 
What are point plots? 
* Pointa show mean of quantitative variable 
* Vertical lines show 95% confidence intervals 

Point plots vs. line plots 
Both show:

* Mean of quantitative variable 
* 95% confidence intervals for the mean 

Differences:
* Line plot has quantitative variable (usually time) on x-axis
* Point plot has categorical variable on x-axis

Point plots vs. bar plots
Both show:
* Mean of quantitative variable 
* 95% confidence intervals for the mean 

import matplotlib.pyplot as plt 
import seaborn as sns 

sns.catplot(x="age",
            y="masculinity_important",
            data=masculinity_data,
            hue="feel_masculine",
            kind="point")
plt.show()


Disconnecting the points 
import matplotlib.pyplot as plt 
import seaborn as sns 

sns.catplot(x="age",
            y="masculinity_important",
            data=masculinity_data,
            hue="feel_masculine",
            kind="point",
            join=False)
 plt.show()
 
 
 Displaying the median 
 import matplotlib.pyplot as plt 
 import seaborn as sns
 
 sns.catplot(x="smoker",
                y="total_bill",
                data=tips,
                kind="point"
                estimator=median)
 plt.show()
 
 Customizing the confidence intervals 
 import matplotlib.pyplot as plt 
 import seaborn as sns 
 
 sns.catplot(x="smoker",
                y="total_bill",
                data=tips,
                kind="point",
                capsize=0.2)
 plt.show()
 
 Turning off confidence intervals
 import matplotlib.pyplot as plt 
 import seaborn as sns
 
 sns.catplot(x="smoker",
            y="total_bill",
            data="point",
            ci=None)
plt.show()
'''


 ###Customizing point plots
 # Add caps to the confidence interval
sns.catplot(x="famrel", y="absences",
			data=student_data,
            kind="point")
        
# Show plot
plt.show()

# Remove the lines joining the points
sns.catplot(x="famrel", y="absences",
			data=student_data,
            kind="point",
            capsize=0.2, join=False)
            
# Show plot
plt.show()


###Point plots with subgroups
# Create a point plot with subgroups
sns.catplot(data=student_data, x="romantic", y="absences", hue="school",kind="point")


# Show plot
plt.show()


# Import median function from numpy
from numpy import median

# Plot the median number of absences instead of the mean
sns.catplot(x="romantic", y="absences",
			data=student_data,
            kind="point",
            hue="school",
            ci=None,
            estimator=median)

# Show plot
plt.show()


'''
Why customize?
Reasons to change style:
* Personal preference 
* Improve readability 
* Guide interpretation

Changing the figure style 
* figure "style" includes background and axes 
* Preset options: "white", "dark", "whitegrid", "darkgrid", "ticks"
* sns.set_style()

Default figure style ("white")
sns.catplot(x="age",
            y="masculinity_important"
            data=masculinity_data,
            hue="feel_masculine",
            kind="point")
plt.show()

Changing the palette 
* Figure "palette" changes the color of the main elements of the plot 
* sns.set_palette()
* Use preset paettes or create a custom paleatte 

Diverging palettes 
"RdBu", "PRGn", "RdBu_r", "PRGn_r"

Example (default paelette)
category_order = ["No answer",
                    "Not at all",
                    "Not very"
                    "Somewhat",
                    "Very"]

sns.catplot(x="how_masculine",
            data=masculinity_data,
            kind="count",
            order=category_order)
plt.show()

sns.set_palette("RdBu")
category_order = ["No answer", "Not at all", "Not very", "Somewhat", "Very"]
sns/catplot(x="how_masculine", data=masculinity_data, kind="count", order="category_order")
plt.show()

Sequential palettes 
"Greys", "Blues", "PuRd", "GnBu"

Custom palettes 
custom_palette = ["red", "green", "orange", "blue", "yellow", "purple"]
sns.set_palette(custom_palette)

Custom palettes 
custom_palette = ['#FBB4AE', '#B3CDE3', '#CCEBC5', '#DECBE4', '#FED9A6', 'FFFFCC', '#E5D8BD', '#FDDAEC', '#F2F2F2']
sns.set_palette(custom_palette)

Changing the scale 
* Figure "context" changes the scale of the plot elements and labels 
* sns.set_context()
* Smallest to larget: "paper", "notebook", "talk", "poster"


Default context: "paper"
sns.catplot(x="age",
            y="masculinity_important",
            data=masculinity_data,
            hue="feel_masculine",
            kind="point")
plt.show()

Larger context: "talk"
sns.set_context("talk")
sns.catplot(x="age",
            y="masculinity_important",
            data=hue"feel_masculine",
            kind="point")
plt.show()        
'''

###Cahnging style and pallete
# Set the style to "whitegrid"
sns.set_style("whitegrid")

# Create a count plot of survey responses
category_order = ["Never", "Rarely", "Sometimes", 
                  "Often", "Always"]

sns.catplot(x="Parents Advice", 
            data=survey_data, 
            kind="count", 
            order=category_order)

# Show plot
plt.show()

# Change the color palette to "RdBu"
sns.set_style("whitegrid")
sns.set_palette("Purples")

# Create a count plot of survey responses
category_order = ["Never", "Rarely", "Sometimes", 
                  "Often", "Always"]

sns.catplot(x="Parents Advice", 
            data=survey_data, 
            kind="count", 
            order=category_order)

# Show plot
plt.show()

# Change the color palette to "RdBu"
sns.set_style("whitegrid")
sns.set_palette("RdBu")

# Create a count plot of survey responses
category_order = ["Never", "Rarely", "Sometimes", 
                  "Often", "Always"]

sns.catplot(x="Parents Advice", 
            data=survey_data, 
            kind="count", 
            order=category_order)

# Show plot
plt.show()


###Changing the scale 
# Set the context to "paper"
sns.set_context("paper")

# Create bar plot
sns.catplot(x="Number of Siblings", y="Feels Lonely",
            data=survey_data, kind="bar")

# Show plot
plt.show()

# Change the context to "poster"
sns.set_context("talk")

# Create bar plot
sns.catplot(x="Number of Siblings", y="Feels Lonely",
            data=survey_data, kind="bar")

# Show plot
plt.show()


# Change the context to "poster"
sns.set_context("poster")

# Create bar plot
sns.catplot(x="Number of Siblings", y="Feels Lonely",
            data=survey_data, kind="bar")

# Show plot
plt.show()


###Using a custom palette 
# Set the style to "darkgrid"
sns.set_style("darkgrid")

# Set a custom color palette
sns.set_palette(["#39A7D0", "#36ADA4"])

# Create the box plot of age distribution by gender
sns.catplot(x="Gender", y="Age", 
            data=survey_data, kind="box")

# Show plot
plt.show()


'''
Adding titles and labels: Part 1
Creating informative visualizations 
FacetGrid vs. AxesSubplot objects 
Seaborn plots create two different types of objects: FacetGrid and AxesSubplot
g = sns.scatterplot(x="height", y="weight", data=df)
type(g)

FacetGrid vs. AxesSubplot objects 
Object Type         Plot Types                          Characteristics 
FacetGrid           relplot(), catplot()                Can create subplots
AxesSubplot         scatterplot(), countplot(), etc.    Only creates a single plot 

Adding a title to FacetGrid 
g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")
g.fig.subtitle("New Title")
plt.show()

Adjusting height of title of FacetGrid 
g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")

g.fig.subtitle("New Title",
                y=1.03)
plt.show()
'''


###FacetGrid vs. AxesSubplots
# Create scatter plot
g = sns.relplot(x="weight", 
                y="horsepower", 
                data=mpg,
                kind="scatter")

# Identify plot type
type_of_g = type(g)

# Print type
print(type_of_g)


# Create scatter plot
g = sns.relplot(x="weight", 
                y="horsepower", 
                data=mpg,
                kind="scatter")

# Add a title "Car Weight vs. Horsepower"
g.fig.suptitle("Car Weight vs. Horsepower")

# Show plot
plt.show()


'''
Adding titles and labels: Part 2
Adding a title to AxesSubplot
FacetGrid
g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")
                
g.fig.subtitle("New Title", y=1.03)

AxesSubplot
g = sns.boxplot(x="Region",
                y="Birthrate",
                data=gdp_data)
                
g.set_title("New Title",
            y=1.03)

Titles for subplots
g = sns.catplot(x="Region",
                y="Birthrate".
                data=gdp_data,
                kind="box",
                col="Group")

Titles for subplots
g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box",
                col="Group")
                
g.fig.suptitle("New Title",
                y=1.03)
                
Titles for Subplots
g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box",
                col="Group")

g.fig.suptitle("New Title", y=1.03)

g.set_titles("This is {col_name}")

Adding axis labels 
g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")
             
g.set(xlabel="New X Label",
      ylabel="New Y Label")
plt.show()

Rotating x-axis tick labels 
g = sns.catplot(x="Region",
                y="Birthrate",
                data=gdp_data,
                kind="box")
plt.xticks(rotation=90)
plt.show()
'''

# Create line plot
g = sns.lineplot(x="model_year", y="mpg_mean", 
                 data=mpg_mean,
                 hue="origin")

# Add a title "Average MPG Over Time"
g.set_title("Average MPG Over Time")

# Show plot
plt.show()


# Create line plot
g = sns.lineplot(x="model_year", y="mpg_mean", 
                 data=mpg_mean,
                 hue="origin")

# Add a title "Average MPG Over Time"
g.set_title("Average MPG Over Time")

# Add x-axis and y-axis labels
g.set(xlabel="Car Model Year",
      ylabel="Average MPG")


# Show plot
plt.show()


# Create point plot
sns.catplot(x="origin", 
            y="acceleration", 
            data=mpg, 
            kind="point", 
            join=False, 
            capsize=0.1)

# Rotate x-tick labels
plt.xticks(rotation=90)

# Show plot
plt.show()


'''Putting it all together 
Relational plots 
* Show the relationship between two quantitative variabels 
* Examples: scatter plots, line plots 
sns.relplot(x="x_variable_name",
            y="y_variable_name",
            data=pandas_df,
            kind="scatter")

Categorical plots
* Show the distribution of a quantitative variable within categories defined 
by a categorical variable
* Examples: bar plots, count plots, box plots, point plots
sns.catplot(x="x_variable_name",
            y="y_variable_name",
            data=pandas_df,
            kind="bar")
            
Adding a third variable (hue)
Setting hue will create subgroups that are displayed as different colors on a single plot 

Adding a third variable (row/col)
Setting row and/or col in relplot() or catplot() will create subgroups that are displayed on 
separate subplots

Customization
* Change the background: sns.set_style()
* Change the main element colors: sns.set_palette()
* Change the scale: sns.set_context()

Adding a title 
Object Type             Plot Types                          How to Add Title 
FaceGrid                relplot(), catplot()                g.fig.suptitle()
AxesSubplot             scatterplot(), countplot(), etc.    g.set_title()

Final touches 
Add x- and y-axis labels:
g.set(xlabel="new x-axis label",
      ylabel="new y-axis label")
      
Rotate x-tick labels:
plt.xticks(rotation=90)
'''

###Box plot with subgroups
# Set palette to "Blues"
sns.set_palette("Blues")

# Adjust to add subgroups based on "Interested in Pets"
g = sns.catplot(x="Gender",
                y="Age", data=survey_data, 
                kind="box", hue="Interested in Pets")

# Set title to "Age of Those Interested in Pets vs. Not"
g.fig.suptitle("Age of Those Interested in Pets vs. Not")

# Show plot
plt.show()


###Bar plot with subgroups and subplots
# Set the figure style to "dark"
sns.set_style("dark")

# Adjust to add subplots per gender
g = sns.catplot(x="Village - town", y="Likes Techno", 
                data=survey_data, kind="bar",
                col="Gender")

# Add title and axis labels
g.fig.suptitle("Percentage of Young People Who Like Techno", y=1.02)
g.set(xlabel="Location of Residence", 
       ylabel="% Who Like Techno")

# Show plot
plt.show()