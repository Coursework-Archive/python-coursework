###SUMMARY
'''
Chapter 1
Diagnose dirty data
Side effects of dirty data
Clean data
Chapter 2
Strings Numeric data
Out of range data 
Out of range dates 
Finding duplicates Treating them 
Chapter 3
Finding inconsistent categories 
Treating them with joins 
Finding inconsistent catefories 
Collapsing them into less 
Unifying formate 
Finding lengths
Chapter 4
Unifying currency formats 
Unifying date formats 
Summing accross rows 
Building aseerts functions 
Finding missind data trating them
Chapter 5
Link datasets where joins don't owrk
By learning about record linkage 
'''


## Numeric data or... ?
# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())

# Print the information of ride_sharing
print(ride_sharing.info())

# Print summary statistics of user_type column
print(ride_sharing['user_type'].describe())

# Convert user_type from integer to category
ride_sharing['user_type_cat'] = ride_sharing['user_type'].astype('category')

# Write an assert statement confirming the change
assert ride_sharing['user_type_cat'].dtype == 'category'

# Print new summary statistics 
print(ride_sharing['user_type_cat'].describe())

# Strip duration of minutes
ride_sharing['duration_trim'] = ride_sharing['duration'].str.strip('minutes')

# Convert duration to integer
ride_sharing['duration_time'] = ride_sharing['duration_trim'].astype('int')

# Write an assert statement making sure of conversion
assert ride_sharing['duration_time'].dtype == 'int'

# Print formed columns and calculate average ride duration 
print(ride_sharing[['duration','duration_trim','duration_time']])
print(ride_sharing['duration_time'].mean())

#Drop values using filtering
movies = movies[movies['avg_rating'] <= 5]
#Drop values using .drop()
movies.drop(movies[movies['avg_rating'] > 5].index, inplace = True)
#Assert results
assert movies[avg_rating].max() <= 5
#Convert avg_rating > 5 to 5
movies.loc[movies['avg_rating'] > 5, 'avg_rating'] = 5
#assert statement
assert movies['avg_rating'].max() <= 5

#Date range example
import datetime as dt
import pandas as pd
# Output data types
user_signups.dtypes
#Convert to DateTime
user_signups['subscriptions_date'] = pd.to_datetime(user_signups['subscription_date'])
#Assert that conversion happened 
assert user_signups['subscription_date'].dtype == 'datetime64[ns]'

today_date = dt.date.today()
#Drop values using filtering
user_signups = user_signups['subscription_date'] < today_date]
#Drop values using .drop()
user_signups.drop(user_signups[user_signups['subscription_date'] > today_date].index, inplace = True)

#Drop values using filtering
user_signups.loc[user_signups['subscription_date'] > today_date, 'subscription_date'] = today_date
#Assert is true
assert user_signups.subscription_date.max().date() <= today_date


#Tire size constraints
# Convert tire_sizes to integer
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('int')

# Set all values above 27 to 27
ride_sharing.loc[ride_sharing['tire_sizes'] > 27, 'tire_sizes'] = 27

# Reconvert tire_sizes back to categorical
ride_sharing['tire_sizes'] = ride_sharing['tire_sizes'].astype('category')

# Print tire size description
print(ride_sharing['tire_sizes'].describe())


# Convert ride_date to datetime
ride_sharing['ride_dt'] = pd.to_datetime(ride_sharing['ride_date'])

# Save today's date
today = dt.date.today()

# Set all in the future to today's date
ride_sharing.loc[ride_sharing['ride_dt'] > today, 'ride_dt'] = today

# Print maximum of ride_dt column
print(ride_sharing['ride_dt'].max())

##How to find duplcates

#Get duplicates across all columns
duplicates = height_weight.duplicated()
print(duplicates)

#How to find duplicate rows?
#subset: List of column names to check for duplication
#keep: Whether to keep first('first'), last('last') or all(False) duplicate values

#Column names to check for duplication
column_names = ['first_name', 'last_name', 'address']
duplicates = height_weight.duplicated(subset = column_names, keep = False)
#Output duplicate values
height_weight[duplicates]
#Output duplicate values
height_weight[duplicates].sort_values(by = 'first_name')

##How to treat duplicate values?
#The .drop_duplicates() method
#subset: List of column names to check for duplication
#keep: whether to keep first('first'), last('last') or all (False) duplicate values
#inplace: Drop duplicate rows directly inside DataFrame without creating new object (True)

#Drop duplicates
height_weight.drop_duplicates(inplace = True)

#Output duplicate values
column_names = ['first_name', 'last_name', 'address']
duplicates = height_weight.duplicated(subset = column_names, keep = False)
height_weight[duplicates].sort_values(by = 'first_name')

##How to treat duplicate values?
#Group by column names and produce statistical summaries
column_names = ['first_name', 'last_name', 'address']
summaries = {'height': 'max', 'weight': 'mean'}
height_weight = height_weight.groupby(by = column_names).agg(summaries).reset_index()

# Find duplicates
duplicates = ride_sharing.duplicated('ride_id', keep = False)

# Sort your duplicated rides
duplicated_rides = ride_sharing[duplicates].sort_values('ride_id')

# Print relevant columns of duplicated_rides
print(duplicated_rides[['ride_id','duration','user_birth_year']])


##Treating Duplicates 
# Drop complete duplicates from ride_sharing
ride_dup = ride_sharing.drop_duplicates()

# Create statistics dictionary for aggregation function
statistics = {'user_birth_year': 'min', 'duration': 'mean'}

# Group by ride_id and compute new statistics
ride_unique = ride_dup.groupby('ride_id').agg(statistics).reset_index()

# Find duplicated values again
duplicates = ride_unique.duplicated(subset = 'ride_id', keep = False)
duplicated_rides = ride_unique[duplicates == True]

# Assert duplicates are processed
assert duplicated_rides.shape[0] == 0


###Categories and membership constraints
#Read study data and print it 
study_data = pd.read_csv('study.csv')
study_data

#Anti-join (what is in A and not in B)
#Inner join (What is in both A and B)

inconsistent_categories = set(study_data['blood_type']).difference(categories['blood_type'])
print(inconsistent_categories)

#Get and print rows with inconsitent categories 
inconsistent_rows = study_data['blood_type'].isin(inconsistent_categories)
study_data[inconsistent_rows]

# drop inconsistent categories and get consistent data 
consistent_data = study_data[~inconsistent_rows]

#Finding consistency 
# Print categories DataFrame
print(categories)

# Print unique values of survey columns in airlines
print('Cleanliness: ', airlines['cleanliness'].unique(), "\n")
print('Safety: ', airlines['safety'].unique(), "\n")
print('Satisfaction: ', airlines['satisfaction'].unique(), "\n")


#Finding consistency 
# Find the cleanliness category in airlines not in categories
cat_clean = set(airlines['cleanliness']).difference(categories['cleanliness'])

# Find rows with that category
cat_clean_rows = airlines['cleanliness'].isin(cat_clean)

# Print rows with inconsistent category
print(airlines[cat_clean_rows])

# Print rows with consistent categories only
print(airlines[~cat_clean_rows])


'''
What type of errors could we have? 
I) Value inconsistency
II) Collapsing too many categories to few 
III) Making sure data is of type 
'''

# Get marriage status column
marriage_status = demographics['marriage_status']
marriage_status.value_counts()

#Capitalize
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.upper()
marriage_status['marriage_status'].value_counts()

#Lowercase
marriage_status['marriage_status'] = marriage_status['marriage_status'].str.lower()
marriage_status['marriage_status'].value_counts()

#Get marriage status column 
marriage_status = demographics['marriage_status']
marriage_status.value_counts()

##Create categories out of data: income_grtoup column from income column 

#Using qcut()
import pandas as pd 
group_names = ['0-200K','200K-500K', '500K+']
demographics['income_group'] = pd.qcut(demographics['household_income'], q = 3, labels = group_names)

#Print income_group column
demographics[['income_group', 'household_income']]

##Collapsing data into categories
#Using cut() - create category ranges and names 
ranges = [0, 200000, 500000, np.inf]
group_names = ['0-200K', '200K-500K', '500K+']

#Create income group column
demographics['income_group'] = pd.cut(demographics['household_income'], bins=ranges, labels=group_names)
demographics[['income_group', 'household_income']]

##Collapsing data into categories 
#Map categoris to fewer ones: reducing categories in categorical column 
#operating_system column is: 'Microsoft', 'MacOS', 'IOS', 'Android', 'Linux'
#operating_system column should become: 'DesktopOS', 'MobileOS'

#Create mapping dictionary and replace 
mapping = {'Microsoft':'DesktopOS', 'MacOS':'DesktopOS', 'Linux':'DesktopOS','IOS':'MobileOS', 'Android':'MobileOS'}
devices['operating_system'] = devices['operating_system'].replace(mapping)
devices['operating_system'].unique()


#Inconsistent categories 
# Print unique values of both columns
print(airlines['dest_region'].unique())
print(airlines['dest_size'].unique())

# Lower dest_region column and then replace "eur" with "europe"
airlines['dest_region'] = airlines['dest_region'].str.lower() 
airlines['dest_region'] = airlines['dest_region'].replace({'eur':'europe'})

# Remove white spaces from `dest_size`
airlines['dest_size'] = airlines['dest_size'].str.strip()

# Verify changes have been effected
print(airlines['dest_region'])
print(airlines['dest_size'])

##Remapping categories 
# Create ranges for categories
label_ranges = [0, 60, 180, np.inf]
label_names = ['short', 'medium', 'long']

# Create wait_type column
airlines['wait_type'] = pd.cut(airlines['wait_min'], bins = label_ranges, 
                                labels = label_names)

# Create mappings and replace
mappings = {'Monday':'weekday', 'Tuesday':'weekday', 'Wednesday': 'weekday', 
            'Thursday': 'weekday', 'Friday': 'weekday', 
            'Saturday': 'weekend', 'Sunday': 'weekend'}

airlines['day_week'] = airlines['day'].replace(mappings)


##Common text data problems: data inconsistency, fixed length violations, typos

phones = pd.read_csv('phones.csv')
print(phones)

#Replace "+" with "00"
phones["Phone number"] = phones["Phone number"].str.replace("+", "00")
phones

#Replace "-" with "00"
phones["Phone number"] = phones["Phone number"].str.replace("-", "")
phones

#Replace phone numbers with lower than 10 digits to NaN
digits = phones['Phone number'].str.len()
phones.loc[digits < 10, "Phone number"] = np.nan
phones

#Find length of each row in phone number column
sanit_check = phone['Phone number'].str.len()
assert sanity_check.min() >= 10

#Assert all numbers do not have "+" or "-"
assert phone['Phone number'].str.contains("+|-").any() == False

#Replace letters with nothing 
phone['Phone number'] = phones['Phone number'].str.replace(r'\D+', '')
phones.head()

##Removing titles and taking names 
# Replace "Dr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Dr.","")

# Replace "Mr." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Mr.","")

# Replace "Miss" with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Miss","")

# Replace "Ms." with empty string ""
airlines['full_name'] = airlines['full_name'].str.replace("Ms.","")

# Assert that full_name has no honorifics
assert airlines['full_name'].str.contains('Ms.|Mr.|Miss|Dr.').any() == False

##Keeping descriptive 
# Store length of each row in survey_response column
resp_length = airlines['survey_response'].str.len()

# Find rows in airlines where resp_length > 40
airlines_survey = airlines[resp_length > 40]

# Assert minimum survey_response length is > 40
assert airlines_survey['survey_response'].str.len().min() > 40

# Print new survey_response column
print(airlines_survey['survey_response'])


##Verifying unit uniformity 
temp_fah = temperatures.loc[temperatures['Temperature'] > 40, 'Temperature']
temp_cels = (temp_fah - 32) * (5/9)
temperatures.loc[temperatures['Temperature'] > 40, 'Temperature'] = temp_cels
#Assert conversion is correct 
assert temperaturs['Temperature'].max() < 40

#Import matplotlib
import matplotlib.pyplot as plt
#Create scatter plot 
plt.scatter(x = 'Date', y = 'Temperature', data = temperatures)
#Create title, xlabel and ylabel 
plt.title('Temperature in celsius March 2019 - NYC')
plt.xlabel('Dates')
plt.ylabel('Temperature in Celsius')
# Show plot
plt.show()

#Treating date data, Attempt to infer format of each date, return NA for rows where conversion failed
birthdays['Birthday'] = pd.to_datetime(birthdays['Birthday']), infer_datetime_format=True, errors='coerce')

birthdays['Birthday'] = birthdays['Birthday'].dt.strftime("%d-%m-%Y")


##Uniform currencies 
# Find values of acct_cur that are equal to 'euro'
acct_eu = banking['acct_cur'] == 'euro'

# Convert acct_amount where it is in euro to dollars
banking.loc[acct_eu, 'acct_amount'] = banking.loc[acct_eu, 'acct_amount'] * 1.1

# Unify acct_cur column by changing 'euro' values to 'dollar'
banking.loc[acct_eu, 'acct_cur'] = 'dollar'

# Assert that only dollar currency remains
assert banking['acct_cur'].unique() == 'dollar'

#Uniform dates
# Print the header of account_opend
print(banking['account_opened'].head())

# Convert account_opened to datetime
banking['account_opened'] = pd.to_datetime(banking['account_opened'],
                                           # Infer datetime format
                                           infer_datetime_format = True,
                                           # Return missing value for error
                                           errors = 'coerce') 

# Get year of account opened
banking['acct_year'] = banking['account_opened'].dt.strftime('%Y')

# Print acct_year
print(banking['acct_year'])


##Cross field validation 
#Example 1
#find instances where the total passengers column is equal to the sum of the classes 
sum_classes = flights[['economy_class', 'business_class', 'first_class']].sum(axis = 1)
passenger_equ = sum_classes == flights['total_passengers']
#Find and filter out rows with inconsistent passenger totals 
inconsistent_pass = flights[~passenger_equ]
consistent_pass = flights[passenger_equ]

#Example 2
import pandas as pd 
import datetime as dt 

#Convert to datetime and get today's date
user['Birthday'] = pd.to_datetime(user['Birthday'])
today = dt.date.today()

#For each row in the Birthday colum, calculate year differences 
age_manual = today.year - year['Birthday'].dt.year

#Find instances where ages match 
age_equ = age_manual == users['Age']
#Find and filter out rows with inconsistent age
inconsistent_age = users[~age_equ]
consistent_age = users[age_equ]

##what to do when we catch inconsistencies?
#drop it 
#set to missing and impute 
#Apply rules from domain knowledge 

##How is your data integrity 
# Store fund columns to sum against
fund_columns = ['fund_A', 'fund_B', 'fund_C', 'fund_D']

# Find rows where fund_columns row sum == inv_amount
inv_equ = banking[fund_columns].sum(axis=1) == banking['inv_amount']

# Store consistent and inconsistent data
consistent_inv = banking[inv_equ]
inconsistent_inv = banking[~inv_equ]

# Store consistent and inconsistent data
print("Number of inconsistent investments: ", inconsistent_inv.shape[0])

# Store today's date and find ages
today = dt.date.today()
ages_manual = today.year - banking['birth_date'].dt.year

# Find rows where age column == ages_manual
age_equ = banking['age'] == ages_manual

# Store consistent and inconsistent data
consistent_ages = banking[age_equ]
inconsistent_ages = banking[~age_equ]

# Store consistent and inconsistent data
print("Number of inconsistent ages: ", inconsistent_ages.shape[0])

##What is missing data
#Return missing values 
airquality.isna()
#Get a summary of missingness 
airquality.isna().sum()

##Missingno
import missingno as msno
import matplotlib.pyplot as plt 
#visualize missingness 
msno.matrix(airquality)
plt.show()

##Airquality example 
#Isolate missing and complete values aside 
missing = airquality[airquality['CO2'].isna()]
complete = airquality[~airquality['CO2'].isna()]

complete.describe()
missing.describe()

#Completness 
sorted_airquality = airquality.sort_values(by = 'Temperature')
msno.matrix(sorted_airquality)
plt.show()

##Missingness types 
#Missing coimpletely random data (MCAR) - no systematic relationships between missing data and other values (Ex. data entry errors when inputting data)
#Missing at random (MAR) - systematic relationships between missing data and other observed values (Ex. Missing ozone data for high temperatures)
#Missing Not at Random (MNAR) - Systematic relationship between missing data and unobserved values (Missing temperature values for high temperatures)

##How to deal with missing data?
#Simple approaches: Drop missing data, Impute with statistical measures (mean, median, mode..)
#More complex approaches: Imputing using an algorithmic approach, impute with machine learning models

##Replacing with statistical measures 
co2_mean = airquality['CO2'].mean()
airquality_imputed = airquality.fillna({'CO2': co2_mean})
airquality_imputed.head()

##Missing investors
# Print number of missing values in banking
print(msno.matrix(banking))

# Visualize missingness matrix
msno.matrix(banking)
plt.show()


# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()

# Isolate missing and non missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]


# Print number of missing values in banking
print(banking.isna().sum())

# Visualize missingness matrix
msno.matrix(banking)
plt.show()

# Isolate missing and non missing values of inv_amount
missing_investors = banking[banking['inv_amount'].isna()]
investors = banking[~banking['inv_amount'].isna()]

# Sort banking by age and visualize
banking_sorted = banking.sort_values('age')
msno.matrix(banking_sorted)
plt.show()


##Follow the money
# Drop missing values of cust_id
banking_fullid = banking.dropna(subset = ['cust_id'])

# Compute estimated acct_amount
acct_imp = banking_fullid['inv_amount'] * 5

# Impute missing acct_amount with corresponding acct_imp
banking_imputed = banking_fullid.fillna({'acct_amount':acct_imp})

# Print number of missing values
print(banking_imputed.isna().sum())


##Minimum edit distance - a way to identify how close two strings are 
#Lets us compare between two strings 
from fuzzywuzzy import fuzz

#Compare reeding vs reading
fuzz.WRatio('Reeding', 'Reading') #Returns a score from 0 to 100 with 100 being an exact match 

#Partial string comparison
fuzz.WRatio('Houston Rockets', 'Rockets')

#Partial string comparison with different order 
fuzz.WRatio('Houston Rockets vs Los Angeles Lakers', 'Lakers vs Rockets')

##Comparison with arrays 
#Import process 
from fuzzywuzzy import process 

#Define string and array of possible match 
string = "Houston Rockets vs Los Angeles Lakers"
choices = pd.Series(['Rockets vs Lakers', 'Lakers vs Rockets', 'Houston vs Los Angeles', 'Heat vs Bulls'])

process.extract(string, choices, limit = 2)

##Replace strings when there is too many variations
#For each correct category
for state in categories['state']:
    #Find potential matches in states with typoes 
    matches = process.extract(state, survey['states'], limit = survey.shape[0])
    #For each potential match 
    for potential_match in matches:
        #If high similarity score
        if potential_match[1] >= 80:
        #Replace typo with correct category
        survey.loc[survey['state'] == potential_match[0], 'state'] = state




#The cutoff point
# Import process from fuzzywuzzy
from fuzzywuzzy import process

# Store the unique values of cuisine_type in unique_types
unique_types = restaurants['cuisine_type'].unique()

# Calculate similarity of 'asian' to all values of unique_types
print(process.extract('asian', unique_types, limit = len(unique_types)))

# Calculate similarity of 'american' to all values of unique_types
print(process.extract('american', unique_types, limit = len(unique_types)))

# Calculate similarity of 'italian' to all values of unique_types
print(process.extract('italian', unique_types, limit = len(unique_types)))


##Remapping categories II
# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract('italian', restaurants['cuisine_type'], limit = len(restaurants['cuisine_type']))

# Inspect the first 5 matches
print(matches[0:5])

# Create a list of matches, comparing 'italian' with the cuisine_type column
matches = process.extract('italian', restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))

# Iterate through the list of matches to italian
for match in matches:
  # Check whether the similarity score is greater than or equal to 80
  if match[1] >= 80:
    # Select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
    restaurants.loc[restaurants['cuisine_type'] == match[0], 'cuisine_type'] = 'italian'
    
# Iterate through categories
for cuisine in categories:  
  # Create a list of matches, comparing cuisine with the cuisine_type column
  matches = process.extract(cuisine, restaurants['cuisine_type'], limit=len(restaurants.cuisine_type))

  # Iterate through the list of matches
  for match in matches:
     # Check whether the similarity score is greater than or equal to 80
    if match[1] >= 80:
      # If it is, select all rows where the cuisine_type is spelled this way, and set them to the correct cuisine
      restaurants.loc[restaurants['cuisine_type'] == match[0]] = cuisine
      
# Inspect the final result
restaurants['cuisine_type'].unique()


##Use blocking when there are big DataFrames and you need to generate pairs between all values 
import recordLinkage

#Create indexing object 
indexer =recordlinkage.Index()

#Generate pairs blocked on state 
indexer.block('state')
pairs = indexer.index(census_A, census_B)

print(pairs)

#Generate the pairs 
pairs = indexer.index(census_A, census_B)

#Create a compare object
compare_cl = recordlinkage.Compare()

#Find exact matches for pairs of date_of_birth and state 
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('state', 'state', label='state')

#Find matches 
potential_matches = compare_cl.compute(pairs, census_A, census_B)
print(potential_matches)
potential_matches[potential_matches.sum(axis = 1) => 2]

#To link or not to link?
# Create an indexer and object and find possible pairs
indexer = recordlinkage.Index()

# Block pairing on cuisine_type
indexer.block('cuisine_type')

# Generate pairs
pairs = indexer.index(restaurants, restaurants_new)


# Create a comparison object
comp_cl = recordlinkage.Compare()

# Find exact matches on city, cuisine_types - 
comp_cl.exact('city', 'city', label='city')
comp_cl.exact('cuisine_type', 'cuisine_type', label='cuisine_type')

# Find similar matches of rest_name
comp_cl.string('rest_name', 'rest_name', label='name', threshold = 0.8) 

# Get potential matches and print
potential_matches = comp_cl.compute(pairs, restaurants, restaurants_new)
print(potential_matches)

##Example 2
#Import recordlinkage and generate full pairs 
import recordlinkage
indexer = recordlinkage.Index()
indexer.block('state')
full_pairs = indexer.index(census_A, census_B)

#Comparison step 
compare_cl = recordlinkage.Compare()
compare_cl.exact('date_of_birth', 'date_of_birth', label='date_of_birth')
compare_cl.exact('state', 'state', label='state')
compare_cl.string('surname', 'surname', threshold=0.85, label='surname')
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')

potential_matches = compare_cl.compute(full_pairs, census_A, census_B)

#Probable matches
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
print(matches)
matches.index

#Get indices from census_B only
duplicate_rows = matches.index.get_level_value(1)
print(census_B_index)

#Finding duplicates in census_B
census_B_duplicates = census_B[census_B.index.isin(duplicate_rows)]

#Finding new rows in census_B
census_B_new = census_B[~census_B.index.isin(duplicate_rows)]

#Link the DataFrames!
full_census = census_A.append(census_B_new)

#Import recordLinkange and generate pairs and compare across columns
''''''
#Generate potential matches
potential_matches = compare_cl.compute(full_pairs, census_A, census_B)

#Isolate matches with matching values for 3 or more columns
matches = potential_matches[potential_matches


##Example 3
#Import recordLinkage and generate full pairs 
import recordlinkage 
indexer = recordlinkage.Index()
indexer.block('state')
full_ pairs = indexer.index(census_A, census_B)

#Comparison step 
compare_cl = recordlinkage.Compare()
compare_cl.exact('date_of_birth', 'date_of_bith', label='date_of_birth')
compare_cl.exact('state', 'state', threshold=0.85, label='surname')
compare_cl.string('surname', 'surname', threshold=0.85, label='surname')
compare_cl.string('address_1', 'address_1', threshold=0.85, label='address_1')

potential_matches = compare_cl.compute(full_pairs, census_A, census_B)

matches = potential_matches[potential_matches.sum(axis = 1) >= 3]
print(matches)

matches.index

#Get indices from census_B only 
duolicate_rows = matches.index.get_level_values(1)
print(census_B_Index)

#Finding duplicates in census_B
census_B_duplicate = census_B[census_B.index.isin(duplicate_rows)]


#Finding new rows in census_B
census_B_new = census_B[~census_B.index.isin)(duplicate_rows)]

#Link the DataFrames!
full_census = census_A.append(census_B_new)

#Import recordlinkage and generates pairs and compare across columns 
''''''
#Generate potential matches 
potential_matches = compare_cl.compute(full_pairs, census_A, census_B)

#Isolate matches with matching values for 3 or more columns 
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]

#Get index for matching census_B rows only 
duplicate_rows = matches.index.get_level_values(1)

#Finding new rows in census_B
census_B_new = census_B[~census_B.index.isin(duplicate_rows)]

#Link the DataFrames!
full_census = census_A.append(census_B_new)


##Linking them together 
# Isolate potential matches with row sum >=3
matches = potential_matches[potential_matches.sum(axis = 1) >= 3]

# Get values of second column index of matches
matching_indices = matches.index.get_level_values(1)

# Subset restaurants_new based on non-duplicate values
non_dup = restaurants_new[~restaurants_new.index.isin(matching_indices)]

# Append non_dup to restaurants
full_restaurants = restaurants.append(non_dup)
print(full_restaurants)
