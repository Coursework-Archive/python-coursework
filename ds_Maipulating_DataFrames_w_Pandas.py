import pandas as pd
df = pd.read_csv('sales.csv', index_col='month')
df

#columne lable in the brakets, row label rihgt braket
df['salt']['Jan']

#loc uses labels
#iloc uses index positions
#left braket row specifier comma column specifier right braket

# Assign the row position of election.loc['Bedford']: x
x = 4
# Assign the column position of election['winner']: y
y = 4

# Print the boolean equivalence
print(election.iloc[x, y] == election.loc['Bedford', 'winner'])

# Import pandas
import pandas as pd

# Read in filename and set the index: election
election = pd.read_csv(filename, index_col='county')

# Create a separate dataframe with the columns ['winner', 'total', 'voters']: results
results = election[['winner', 'total', 'voters']]

# Print the output of results.head()
print(results.head())


####Slicing data frames
#basic index for this picks a column by default
df['eggs']
type(df['eggs'])
#pandas.core.series.Series

df['eggs'][1:4] # Part of the eggs column, rows 1, 2 and 3
df['eggs'][4] #the 4th wow of the eggs column

df.loc[:, 'eggs':'salt'] # first colon selects all rows, the next slice selects those labeled columns.
df.loc['Jan':'Apr',:] #selects the labeled rows and selects all columns
df.loc['Mar':'May', 'salt':'spam'] #selects the labeled rows and columns

df.iloc[2:5, 1:] #selects rows 2,3 and 4, selects 1 to the end of colomns

#You can use lists in place of slices using .loc
df.loc['Jan':'May', ['eggs', 'spam']] #selects the labeled row and a list of two columns

df.iloc[[0,4,5], 0:2] #a list of arow and a slice of column

df['eggs'] #yeilds a Series by column name

df[['eggs']] #returns the DataFram w/single column

#not all operations are shared between dataframs and series, a series is always one dimension of labeled data
# Slice the row labels 'Perry' to 'Potter': p_counties
p_counties = election.loc['Perry':'Potter']

# Print the p_counties DataFrame
print(p_counties)

# Slice the row labels 'Potter' to 'Perry' in reverse order: p_counties_rev
p_counties_rev = election.loc['Potter':'Perry':-1]

# Print the p_counties_rev DataFrame
print(p_counties_rev)

# Slice the columns from the starting column to 'Obama': left_columns
left_columns = election.loc[:,:'Obama']

# Print the output of left_columns.head()
print(left_columns.head())

# Slice the columns from 'Obama' to 'winner': middle_columns
middle_columns = election.loc[:,'Obama':'winner']

# Print the output of middle_columns.head()
print(middle_columns.head())

# Slice the columns from 'Romney' to the end: 'right_columns'
right_columns = election.loc[:,'Romney':]

# Print the output of right_columns.head()
print(right_columns.head())

# Create the list of row labels: rows
rows = ['Philadelphia', 'Centre', 'Fulton']

# Create the list of column labels: cols
cols = ['winner', 'Obama', 'Romney']

# Create the new DataFrame: three_counties
three_counties = election.loc[rows, cols]

# Print the three_counties DataFrame
print(three_counties)

#Filtering can also be used to select data not based on labels or positions
#Based on properties of interes in the data itself
df.salt > 60 #this returns a Boolean Series and is called a filter
df[df.salt > 60] #a filter can be used with brakets as a logical expession
enough_salt_sold = df.salt > 60
df[enough_salt_sold] #or assigne to another variable

df[(df.salt >= 50) & (df.eggs < 200)] # using & to meet both conditions

df[(df.salt >= 50) | (df.eggs < 200)] #Either condition OR

df2 = df.copy() #can be used to not include zeros or NaN
df2['bacon'] = [0, 0, 50, 60, 70, 80]
df2

df2.loc[:, df2.all()] #select columns with all nonzeros
df2.loc[:, df2.any()] #elects all columns with any nonzeros
df.loc[:, df.isnull().any()] #which row/columns contain NaN
df.loc[:, df.notnull().all()] #which row/columns contain values
df.dropna(how='any') #drop rows with any NaN

# Create the boolean array: high_turnout
high_turnout = election.turnout > 70

# Filter the election DataFrame with the high_turnout array: high_turnout_df
high_turnout_df = election[high_turnout]

# Print the high_turnout_results DataFrame
print(high_turnout_df)

# Import numpy
import numpy as np

# Create the boolean array: too_close
too_close = election.margin < 1

# Assign np.nan to the 'winner' column where the results were too close to call
election.loc[too_close, 'winner'] = np.nan

# Print the output of election.info()
print(election.info())

# Select the 'age' and 'cabin' columns: df
df = titanic.loc[:,['age','cabin']]

# Print the shape of df
print(df.shape)

# Drop rows in df with how='any' and print the shape
print(df.dropna(how='any').shape)

# Drop rows in df with how='all' and print the shape
print(df.dropna(how='all').shape)

# Drop columns in titanic with less than 1000 non-missing values
print(titanic.dropna(thresh=1000, axis='columns').info())

df.flooriv(12) # Convert to dozens unit

#using vectorized or element based computation
import numpy as np
np.floor_divide(df, 12) # Convert to dozen unit

#Can create a python function
def dozens(n):
    return n // 12

df.apply(dozens) #convert t odozens

#Can acheive the same result using the lambda key word
df.apply(lambda n: n // 12)

#Can create a new column storing functions
df['dozens_of_eggs'] = df.eggs.flordiv(12)
df

df.index #special kind of Series containing strings

df.index = df.index.str.upper() #makes the index all uppercase

#for the index there is no apply method for the index you must use map
df.index = df.index.map(str.lower)
df

#plus signs can be used with Series and DataFrames
df['salty_eggs'] = df.salt + df.dozens_of_eggs
df

# Write a function to convert degrees Fahrenheit to degrees Celsius: to_celsius
def to_celsius(F):
    return 5/9*(F - 32)

# Apply the function over 'Mean TemperatureF' and 'Mean Dew PointF': df_celsius
df_celsius = weather[['Mean TemperatureF', 'Mean Dew PointF']].apply(to_celsius)

# Reassign the column labels of df_celsius
df_celsius.columns = ['Mean TemperatureC', 'Mean Dew PointC']

# Print the output of df_celsius.head()
print(df_celsius.head())

# Create the dictionary: red_vs_blue
red_vs_blue = {'Obama':'blue', 'Romney':'red'}

# Use the dictionary to map the 'winner' column to the new column: election['color']
election['color'] = election['winner'].map(red_vs_blue)

# Print the output of election.head()
print(election.head())

# Import zscore from scipy.stats
from scipy.stats import zscore

# Call zscore with election['turnout'] as input: turnout_zscore
turnout_zscore = zscore(election['turnout'])

# Print the type of turnout_zscore
print(type(turnout_zscore))

# Assign turnout_zscore to a new column: election['turnout_zscore']
election['turnout_zscore'] = turnout_zscore

# Print the output of election.head()
print(election.head())


####Advanced Indexing
#Indexes: Squence of labels
##Immutable (Like dictionary keys)
##Homogeneous in data type (Like NumPy arrays)
#Series: 1D array with index
#DataFrams: 2D array with Series as columns

#Creating a Series
import pandas as pd
prices = [10.70, 10.86, 10.74, 10.71, 10.79]
shares = pd.Series(prices)
print(shares)

#Creating an Index
days = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri']
shares = pd.Series(prices, index=days)
print(shares)

print(shares.index[2])
#Wed

print(shares.index[:2])
#Index(['Mon', 'Tue'], dtype='object')

print(shares.index[-2:])
#Index(['Thu', 'Fri'], dtype='object')

print(shares.index.name)
#None

#Modifying the index name
shares.index.name = 'weekday'
print(shares)

#indexes can be reassigned by overwriting it all at once, cannot change individual labels
shares.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(shares)

unemployment = pd.read_csv('Unemployent.csv')
unemployment.head()

#Assigning the index
unemployment.index = unemployment['Zip']
unemployment.head()

#Index objects and labeled data
unemployment = pd.read_csv('Unemployment.csv', index_col='Zip')
unemployment.head()

# Create the list of new indexes: new_idx
new_idx = [i.upper() for i in sales.index]

# Assign new_idx to sales.index
sales.index = new_idx

# Print the sales DataFrame
print(sales)

# Assign the string 'MONTHS' to sales.index.name
sales.index.name = 'MONTHS'

# Print the sales DataFrame
print(sales)

# Assign the string 'PRODUCTS' to sales.columns.name
sales.columns.name = 'PRODUCTS'

# Print the sales dataframe again
print(sales)

# Generate the list of months: months
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']

# Assign months to sales.index
sales.index = months

# Print the modified sales DataFrame
print(sales)


##Hierarchical Indexing
import pandas as pd
stocks = pd.read_csv('datasets/stocks.csv')
print(stocks)

#setting an index using a tuple
stocks = stocks.set_index(['Symbol', 'Date'])
print(stocks)

#The index can be sorted
stocks = stocks.sort_index()
print(stocks)

#Hierarhical indexing; indexing an individual row
stocks.loc[('CSCO', '2016-10-04')]

stocks.loc[('CSCO', '2016-10-04'), 'Volume']

#Fancy indexing (outermost index)
stocks.loc[(['AAPL', 'MSFT'], '2016-10-05'), :]

#extracting only the closed columns on those days
stocks.loc[(['AAPL', 'MSFT'], '2016-10-05'), 'Close']

#extracting data from the innermost row
stocks.loc[('CSCO', ['2016-10-05', '2016-10-03']), :]

#Slicing with hierarchical index
stocks.loc[(slice(None), slice('2016-10-03', '2016-10-04')),:]


# Print sales.loc[['CA', 'TX']]
print(sales.loc[['CA', 'TX']])

# Print sales['CA':'TX']
print(sales['CA':'TX'])

# Set the index to be the columns ['state', 'month']: sales
sales = sales.set_index(['state', 'month'])

# Sort the MultiIndex: sales
sales = sales.sort_index()

# Print the sales DataFrame
print(sales)

# Set the index to the column 'state': sales
sales = sales.set_index(['state'])

# Print the sales DataFrame
print(sales)

# Access the data from 'NY'
print(sales.loc['NY'])

# Look up data for NY in month 1 in sales: NY_month1
NY_month1 = sales.loc['NY', 1]

# Look up data for CA and TX in month 2: CA_TX_month2
CA_TX_month2 = sales.loc[(['CA','TX'], 2),:]

# Access the inner month index and look up data for all states in month 2: all_month2
all_month2 = sales.loc[(['CA','NY','TX'], 2),:]


###Pivoting DataFrames
import pandas as pd
trails = pd.read_csv('trails_01.csv')
print(trails)

#Reshaping by pivoting
trails.pivot(index='treatment',
             columns='gender',
             values='response')

# Pivot the users DataFrame: visitors_pivot
visitors_pivot = users.pivot(index='weekday', columns='city', values='visitors')

# Print the pivoted DataFrame
print(visitors_pivot)

# Pivot users with signups indexed by weekday and city: signups_pivot
signups_pivot = users.pivot(index='weekday', columns='city', values='signups')

# Print signups_pivot
print(signups_pivot)

# Pivot users pivoted by both signups and visitors: pivot
pivot = users.pivot(index='weekday', columns='city')

# Print the pivoted DataFrame
print(pivot)

#Stacking & unstacking DataFrames

#You can create a multi-level hierarchical index using .set_index()

#Unstacking a multi-indox DataFrame
trials.unstack(level='gender')

#Can also unstack using the level indexing in a zero based index
trials.unstack(level=1)

#stacking DataFrames
trials_by_gender.stack(level='gender')

#swapping levels
stacked.swaplevel(0, 1)

#after swapping the index remains unsorted, you can sort the index
swapped.sort_index()

# Unstack users by 'weekday': byweekday
byweekday = users.unstack(level='weekday')

# Print the byweekday DataFrame
print(byweekday)

# Stack byweekday by 'weekday' and print it
print(byweekday.stack(level='weekday'))

# Unstack users by 'city': bycity
bycity = users.unstack(level='city')

# Print the bycity DataFrame
print(bycity)

# Stack bycity by 'city' and print it
print(bycity.stack(level='city'))

# Stack 'city' back into the index of bycity: newusers
newusers = bycity.stack(level='city')

# Swap the levels of the index of newusers: newusers
newusers = newusers.swaplevel(0, 1)

# Print newusers and verify that the index is not sorted
print(newusers)

# Sort the index of newusers: newusers
newusers = newusers.sort_index()

# Print newusers and verify that the index is now sorted
print(newusers)

# Verify that the new DataFrame is equal to the original
print(newusers.equals(users))


#Melting DataFrames
new_trials = pd.read_csv('trials_02.csv')
pd.melt(new_trials)

#if column identifiers disappear after melting you can rename them
pd.melt(new_trails, id_vars=['treatment'])

#can also explicitly list which columns to convert to values
pd.melt(new_trials, id_vars=['treatment'], value_vars=['F','M'])

#you can provide more descriptive names with var_name and value_name
pd.melt(new_trials, id_vars=['treatment'], var_name='gender', value_name='response')

# Reset the index: visitors_by_city_weekday
visitors_by_city_weekday = visitors_by_city_weekday.reset_index()

# Print visitors_by_city_weekday
print(visitors_by_city_weekday)

# Melt visitors_by_city_weekday: visitors
visitors = pd.melt(visitors_by_city_weekday, id_vars=['weekday'], value_name='visitors')

# Print visitors
print(visitors)

# Melt users: skinny
skinny = pd.melt(users, id_vars=['weekday', 'city'])

# Print skinny
print(skinny)

# Set the new index: users_idx
users_idx = users.set_index(['city', 'weekday'])

# Print the users_idx DataFrame
print(users_idx)

# Obtain the key-value pairs: kv_pairs
kv_pairs = pd.melt(users_idx, col_level=0)

# Print the key-value pairs
print(kv_pairs)

#pivot table, pivot doesn't always work repeatedd pairs could make it impossible
more_trials.pivot_table(index='treatment', columns='gender', values='response', aggfunc='count')

# Create the DataFrame with the appropriate pivot table: by_city_day
by_city_day = users.pivot_table(index='weekday', columns='city')

# Print by_city_day
print(by_city_day)

# Use a pivot table to display the count of each column: count_by_weekday1
count_by_weekday1 = users.pivot_table(index='weekday',aggfunc='count')

# Print count_by_weekday
print(count_by_weekday1)

# Replace 'aggfunc='count'' with 'aggfunc=len': count_by_weekday2
count_by_weekday2 = users.pivot_table(index='weekday',aggfunc=len)

# Verify that the same result is obtained
print('==========================================')
print(count_by_weekday1.equals(count_by_weekday2))

# Create the DataFrame with the appropriate pivot table: signups_and_visitors
signups_and_visitors = users.pivot_table(index='weekday',aggfunc=sum)

# Print signups_and_visitors
print(signups_and_visitors)

# Add in the margins: signups_and_visitors_total
signups_and_visitors_total = users.pivot_table(index='weekday', aggfunc=sum, margins=True)

# Print signups_and_visitors_total
print(signups_and_visitors_total)

###Categoricals and groupby
sales.groupby('weekday').count()

#groupby works with the following statistical functions
#mean(), std(), sum(), first(), last(), min(), max()
#Total amount of bread sold each day can be found with
sales.groupby('weekday')['bread'].sum()

#multiple columns
sales.groupby('weekday')[['bread','butter']].sum()

#multi-level index
sales.groupby(['city','weekday']).mean()

#list of customers
customers = pd.Series(['Dave','Alice','Bob','Alive'])
customers

sales.groupby(customers)['bread'].sum()

#categorical data
sales['weekday'].unique()

#value counts
sales['weekday'] = sales['weekday'].astype('category')
sales['weekday']

#categorical data advantage
#uses less memory
#speeds up operaions like groupby()

# Group titanic by 'pclass'
by_class = titanic.groupby('pclass')

# Aggregate 'survived' column of by_class by count
count_by_class = by_class['survived'].count()

# Print count_by_class
print(count_by_class)

# Group titanic by 'embarked' and 'pclass'
by_mult = titanic.groupby(['embarked','pclass'])

# Aggregate 'survived' column of by_mult by count
count_mult = by_mult['survived'].count()

# Print count_mult
print(count_mult)

# Read life_fname into a DataFrame: life
life = pd.read_csv(life_fname, index_col='Country')

# Read regions_fname into a DataFrame: regions
regions = pd.read_csv(regions_fname, index_col='Country')

# Group life by regions['region']: life_by_region
life_by_region = life.groupby(regions['region'])

# Print the mean over the '2010' column of life_by_region
print(life_by_region['2010'].mean())


###Groupby and aggregation
#grouping by a DataFrame and aggregate based on 1 or more columns
#group the maximum amount of bread or butter for all cities
sales.groupby('city')[['bread','butter']].max()

#Multiple aggregations can be performed
sales.groupby('city')[['bread','butter']].agg(['max','sum'])

#the agg method can be used in several different ways: sum, mean and count can be
#passed

#custom aggregation can also be used with Series instead of DataFrames
def data_range(series):
    return series.max() - series.min()

#Additional custom aggregations
sales.groupby('weekday')[['bread', 'butter']].agg(data_range)

#Custom aggregation accepts a dictionary input too
sales.groupby(customers)[['bread','butter']].agg({'bread':'sum', 'butter':data_range})

# Group titanic by 'pclass': by_class
by_class = titanic.groupby('pclass')

# Select 'age' and 'fare'
by_class_sub = by_class[['age','fare']]

# Aggregate by_class_sub by 'max' and 'median': aggregated
aggregated = by_class_sub.agg(['max','median'])

# Print the maximum age in each class
print(aggregated.loc[:, ('age','max')])

# Print the median fare in each class
print(aggregated.loc[:, ('fare','median')])

# Read the CSV file into a DataFrame and sort the index: gapminder
gapminder = pd.read_csv('gapminder.csv', index_col=['Year','region','Country']).sort_index()

# Group gapminder by 'Year' and 'region': by_year_region
by_year_region = gapminder.groupby(level=['Year', 'region'])

# Define the function to compute spread: spread
def spread(series):
    return series.max() - series.min()

# Create the dictionary: aggregator
aggregator = {'population':'sum', 'child_mortality':'mean', 'gdp':spread}

# Aggregate by_year_region using the dictionary: aggregated
aggregated = by_year_region.agg(aggregator)

# Print the last 6 entries of aggregated
print(aggregated.tail(6))

# Read file: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Create a groupby object: by_day
by_day = sales.groupby(sales.index.strftime('%a'))

# Create sum: units_sum
units_sum = by_day['Units'].sum()

# Print units_sum
print(units_sum)

##Groupby and transformation
#The z-score
def zscore(series):
    return (series.mean() / series.std())

zscore(auto['mpg']).head()

#MPG z-score by year
auto.groupby('yr')['mpg'].transform(zscore).head()

#Apply transformation and aggregation
auto.groupby('yr').apply(zscore_with_year_and_name).head()

#Function zed score with year and name
def zscore_with_year_and_name(group):
    df = pd.DataFrame(
        {'mpg':zscore(group['mpg']),
         'year':group['yr'],
         'name':group['name']})
    return df

auto.groupby('yr').apply(zscore_with_year_and_name).head()

# Import zscore
from scipy.stats import zscore

# Group gapminder_2010: standardized
standardized = gapminder_2010.groupby('region')['life','fertility'].transform(zscore)

# Construct a Boolean Series to identify outliers: outliers
outliers = (standardized['life'] < -3) | (standardized['fertility'] > 3)

# Filter gapminder_2010 by the outliers: gm_outliers
gm_outliers = gapminder_2010.loc[outliers]

# Print gm_outliers
print(gm_outliers)

# Create a groupby object: by_sex_class
by_sex_class = titanic.groupby(['sex','pclass'])

# Write a function that imputes median
def impute_median(series):
    return series.fillna(series.median())

# Impute age and assign to titanic['age']
titanic.age = by_sex_class['age'].transform(impute_median)

# Print the output of titanic.tail(10)
print(titanic.tail(10))

# Group gapminder_2010 by 'region': regional
regional = gapminder_2010.groupby('region')

# Apply the disparity function on regional: reg_disp
reg_disp = regional.apply(disparity)

# Print the disparity of 'United States', 'United Kingdom', and 'China'
print(reg_disp.loc[['United States','United Kingdom','China']])

#Filtering the groups to calculated based on label
splitting = auto.groupby('yr')
type(splitting)

#groupby oject: iteration and filtering
for group_name, group in splitting:
    avg = group.loc[group['name'].str.contains('chevrolet'), 'mpg'].mean()
    pring(group_name, avg)

#the loop can be rewritten as a dictionary comprehension
chevy_means = {year:group.loc[group['name'].str.contains('chevrolet'),'mpg'].mean()
               for year, group in splitting}

pd.Series(chevy_means)

#we can perform a one to all comparison
chevy = auto['name'].str.contains('chevrolet')
auto.groupby(['yr', chevy])['mpg'].mean()

# Create a groupby object using titanic over the 'sex' column: by_sex
by_sex = titanic.groupby('sex')

# Call by_sex.apply with the function c_deck_survival
c_surv_by_sex = by_sex.apply(c_deck_survival)

# Print the survival rates
print(c_surv_by_sex)

# Read the CSV file into a DataFrame: sales
sales = pd.read_csv('sales.csv', index_col='Date', parse_dates=True)

# Group sales by 'Company': by_company
by_company = sales.groupby('Company')

# Compute the sum of the 'Units' of by_company: by_com_sum
by_com_sum = by_company['Units'].sum()
print(by_com_sum)

# Filter 'Units' where the sum is > 35: by_com_filt
by_com_filt = by_company.filter(lambda g:g['Units'].sum() > 35)
print(by_com_filt)


###Case Stuy: olympic medals

# Select the 'NOC' column of medals: country_names
country_names = medals['NOC']

# Count the number of medals won by each country: medal_counts
medal_counts = country_names.value_counts()

# Print top 15 countries ranked by medals
print(medal_counts.head(15))

# Construct the pivot table: counted
counted = medals.pivot_table(index='NOC', columns='Medal', values='Athlete', aggfunc='count')

# Create the new column: counted['totals']
counted['totals'] = counted.sum(axis='columns')

# Sort counted by the 'totals' column
counted = counted.sort_values('totals', ascending=False)

# Print the top 15 rows of counted
print(counted.head(15))

##Understanding the column labels


# Select columns: ev_gen
ev_gen = medals[['Event_gender','Gender']]

# Drop duplicate pairs: ev_gen_uniques
ev_gen_uniques = ev_gen.drop_duplicates()

# Print ev_gen_uniques
print(ev_gen_uniques)

# Group medals by the two columns: medals_by_gender
medals_by_gender = medals.groupby(['Event_gender', 'Gender'])

# Create a DataFrame with a group count: medal_count_by_gender
medal_count_by_gender = medals_by_gender.count()

# Print medal_count_by_gender
print(medal_count_by_gender)

# Create the Boolean Series: sus
sus = (medals.Event_gender == 'W') & (medals.Gender == 'Men')

# Create a DataFrame with the suspicious row: suspect
suspect = medals[sus]

# Print suspect
print(suspect)

##Constructing alternative coountry rankings
medals['Sport'].unique() #42 distinct events

#Two new DataFrame methods
#idxmax(): Row or column label where maximum value is located
#idmin(): Row or column label where minimum value is located

# Group medals by 'NOC': country_grouped
country_grouped = medals.groupby('NOC')

# Compute the number of distinct sports in which each country won medals: Nsports
Nsports = country_grouped['Sport'].nunique()

# Sort the values of Nsports in descending order
Nsports = Nsports.sort_values(ascending=False)

# Print the top 15 rows of Nsports
print(Nsports.head(15))

# Create a Boolean Series that is True when 'Edition' is between 1952 and 1988: during_cold_war
during_cold_war = (medals['Edition'] >= 1952) & (medals['Edition'] <= 1988)

# Extract rows for which 'NOC' is either 'USA' or 'URS': is_usa_urs
is_usa_urs = medals.NOC.isin(['USA', 'URS'])

# Use during_cold_war and is_usa_urs to create the DataFrame: cold_war_medals
cold_war_medals = medals.loc[during_cold_war & is_usa_urs]

# Group cold_war_medals by 'NOC'
country_grouped = cold_war_medals.groupby('NOC')

# Create Nsports
Nsports = country_grouped['Sport'].nunique().sort_values(ascending=False)

# Print Nsports
print(Nsports)


# Create the pivot table: medals_won_by_country
medals_won_by_country = medals.pivot_table(index='Edition', columns='NOC', values='Athlete', aggfunc='count')

# Slice medals_won_by_country: cold_war_usa_urs_medals
cold_war_usa_urs_medals = medals_won_by_country.loc[1952:1988, ['USA','URS']]

# Create most_medals
most_medals = cold_war_usa_urs_medals.idxmax(axis='columns')

# Print most_medals.value_counts()
print(most_medals.value_counts())

#Reshaping:plotting DataFrames
#unstacking the data is also known as reshaping
# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by ['Edition', 'Medal'] and aggregate over 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Plot the DataFrame usa_medals_by_year
usa_medals_by_year.plot()
plt.show()

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()

# Redefine 'Medal' as an ordered categorical
medals.Medal = pd.Categorical(values=medals.Medal, categories=['Bronze', 'Silver', 'Gold'], ordered=True)

# Create the DataFrame: usa
usa = medals[medals.NOC == 'USA']

# Group usa by 'Edition', 'Medal', and 'Athlete'
usa_medals_by_year = usa.groupby(['Edition', 'Medal'])['Athlete'].count()

# Reshape usa_medals_by_year by unstacking
usa_medals_by_year = usa_medals_by_year.unstack(level='Medal')

# Create an area plot of usa_medals_by_year
usa_medals_by_year.plot.area()
plt.show()