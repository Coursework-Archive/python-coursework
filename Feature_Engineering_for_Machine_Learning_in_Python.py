'''
Why generate features?
Feature engineering is the act of taking raw data and extracting feature from it that are 
suitable for tasks like machine learning. Wwhen we talk about features we are referring to
the information stored in the columns of these tables 

Most machine learning algorithms require their input data to be represented as a vector
or a matrix, and many assime that the data is distributed normally.  

Different types of data 
* Continuous: either integers (or whole numbers) or floats (decimals)
* Categorical: one of a limited set of values, e.g gender, country of birth 
* Ordinal: ranked values, often with no detail of distance between them 
* Boolean: True/False values 
* Datetime: dates and times 

import pandas as pd 
df = pd.read_csv(path_to_csv_file)
print(df.head())

print(df.types)

only_ints = df.select_dtypes(include=['int'])
print(only_ints.columns)
'''

###Getting to know your data 
# Import pandas
import pandas as pd

# Import so_survey_csv into so_survey_df
so_survey_df = pd.read_csv(so_survey_csv)

# Print the first five rows of the DataFrame
print(so_survey_df.head(5))

# Print the data type of each column
print(so_survey_df.dtypes)


###Selecting specific data types 
# Create subset of only the numeric columns
so_numeric_df = so_survey_df.select_dtypes(include=['int','float'])

# Print the column names contained in so_survey_df_num
print(so_numeric_df.columns)


'''
Dealing with Categorical Variables 

To get from qualitative inputs to quantitative features, Firstly you cannot apply some 
number to the category because that would imply some form of ordering. Instead values 
can be encoded by creating additional binary feaures corresponding to whether each value
was picked or not as shown in the table on the right 

Encoding categorical features 
* One-hot encoding 
* Dummy encoding 

Pendas uses one-hot encoding when you use the get_dummies function 

pd.get_dummies(df, columns=['Country'], prefix='C') - n features for n categories 

pd.get_dummies(df, columns=['Country'], drop_first=True, prefix='C') - n-1 features for n categories

One-hot vs. dummies 
* One-hot encoding: Explainable features 
* Dummy encoding: Necessary information withouot duplication 


In the case where there are a high number of different values in a column, you may 
want to only create columns for the most common values. You can check the number of 
occurences of different categories in a column using the value_counts method on a specific 
column 

Limiting your columns 
mask = df['Country'].isin(counts[counts < 5].index)
df['Country'][mask] = 'Other'
'''


###One-hot encoding and dummy variables 
# Create dummy variables for the Country column
dummy = pd.get_dummies(so_survey_df, columns=['Country'], drop_first=True, prefix='DM')

# Print the columns names
print(dummy.columns)

###Dealing with uncommon categories 
# Create a series out of the Country column
countries = so_survey_df.Country

# Get the counts of each category
country_counts = countries.value_counts()

# Print the count values for each category
print(country_counts)

# Create a mask for only categories that occur less than 10 times
mask = countries.isin(country_counts[country_counts < 10].index)

# Print the top 5 rows in the mask series
print(mask.head(5))

# Label all other categories as Other
countries[mask] = 'Other'

# Print the updated category counts
print(countries.value_counts())

'''
Numeric variables 
Binarizing numeric variables 
df['Binary_Violation'] = 0
df.loc[df['Number_of_Violations'] > 0,
        'Binary_Violation'] = 1

Binning nmeric variables 
import numpy as np 
df['Binned_Group'] = pd.cut(
    df['Number_of_Violations'],
    bins=[-np.inf, 0, 2, np.inf]),
    labels=[1,2,3]
    )
'''

###Binarizing columns 
# Create the Paid_Job column filled with zeros
so_survey_df['Paid_Job'] = 0

# Replace all the Paid_Job values where ConvertedSalary is > 0
so_survey_df.loc[so_survey_df['ConvertedSalary'] > 0, 'Paid_Job'] = 1

# Print the first five rows of the columns
print(so_survey_df[['Paid_Job', 'ConvertedSalary']].head())


###Binning values 
# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=5)

# Print the first 5 rows of the equal_binned column
print(so_survey_df[['equal_binned', 'ConvertedSalary']].head())


'''
Why do missing values exist?
* Data not being collected properly 
* Collection and management errors 
* Data intentionally beng omitted 
* Could be created due to transformations of the data 

Why do we care?
* Some models cannot work with missing data (Nulls/NaNs)
* Missing data may be a sign of a wider data issue 
* Missing data can be a useful feature 
print(df.info())
print(df.isnull())
print(df['StackOverflowJobsRecomend'].isnull().sum())
print(df.notnull())
'''

###How sparse is my data 
# Subset the DataFrame
sub_df = so_survey_df[['Age','Gender']]

# Print the number of non-missing values
print(sub_df.notnull().sum())

###Finding the missing values 
# Print the top 10 entries of the DataFrame
print(sub_df.head(10))

# Print the locations of the missing values
print(sub_df.head(10).isnull())

# Print the locations of the non-missing values
print(sub_df.head(10).notnull())


'''
Dealing with missing values (I)

Listwise delection in Python 
# Drop all rows with at least one missing values 
df.dropna(how='any')

# Drop rows with missing values in a specific column
df.dropna(subset=['VersionControl'])

Issues with delection 
* It deletes valid data points
* Relies on randomness 
* Reduce information 

Replacing with strings 
# Replace missing values in a specific column with a given string 
df['VersionControl'].fillna(value='None Given', inplace=True)

# Record where the values are not missing 
df['SalaryGiven'] = df['ConvertedSalary'].notnull()

# Drop a specific column
df.drop(columns=['ConvertedSalary'])
'''

###Likewise deletion
# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna(how='any')

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)

# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols = so_survey_df.dropna(how='any', axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)

# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)

###Replace missing values with constant 
# Print the count of occurrences
print(so_survey_df['Gender'].value_counts())

# Replace missing values
so_survey_df['Gender'].fillna(value='Not Given', inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())# Replace missing values
so_survey_df['Gender'].fillna(value='Not Given', inplace=True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())

'''
Fill continuous missing values 
Deleting missing values 
* Can't delete rows missing values in the test set 

What else can you do?
* Cateforical columns: Replace missing values with the most common 
occurring value or with a string that flags missing values such as 'None'
* Numeric columns: Replace missing values with a suitable value 

Measures of central tendency 
* Mean 
* Median

Calculating the measures of central tendency 
print(df['ConvertedSalary'].mean())
print(df['ConvertedSalary'].median())

Fill the missing values 
df['ConvertedSalary'] = df['ConvertedSalary'].fillna(df['ConvertedSalary'].mean())

df['ConvertedSalary'] = df['ConvertedSalary'].astype('int64')

Rounding values 
df['ConvertedSalary'] = df['ConvertedSalary'].fillna(round(df['ConvertedSalary'].mean()))
'''

###Filling continuous missing values 
# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head(5))

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df['StackOverflowJobsRecommend'].head())

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())

'''
Dealing with other data issues 
Bad characteers 

Dealing with bad characters 
df['RawSalary'] = df['RawSalary'].str.replace(',', '')

Chaining methods 
df['column_names'] = df['column_name'].method1()
df['column_names'] = df['column_name'].method2()
df['column_names'] = df['column_name'].method3()

Same as:
df['column_name'] = df['column_name'].method1().method2().method3()

'''

###Dealing with stray characters (I)
# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')

# Remove the dollar signs in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$', '')


###Dealing with stray characters (II)
# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isna()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])

###Dealing with stray characters (II)
# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£','')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print(so_survey_df['RawSalary'])

###Method chaining 
# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
                              .str.replace(',','')\
                              .str.replace('$','')\
                              .str.replace('£','')\
                              .astype('float')
'''
Data distributions 
All models require your features to be on the same scale 

Almost all models besides tree based models assume that your data is normally 
distributed. Normal distributions follow a bell shape like shown here, the main 
characteristics of a normal distribution is that 68.27% of the data lies 
within 1 standard of the mean, 95.45% lies within 2 standard deviations from the 
mean and 99.73% 

Observing your data 
import matplotlib as plt

df.hist()
plt.show()


Delving deeper with box plots
"Minimum"
(Q1 - 1.5 IQR)


Interquartile Range (IQR)
*******************************************
Q1
(25th Percentile)

Q3
(75th Percentile)
*******************************************

"Maximum"
(Q3 + 1.5 IQR)

Box plot in pandas
df[['column_1']].boxplot()
plt.show()

import seaborn as sns
sns.pairplot(df)

df.describe()
'''

###What does your data look like? (I)
# Create a histogram
so_numeric_df.hist()
plt.show()

# Create a boxplot of two columns
so_numeric_df[['Age', 'Years Experience']].boxplot()
plt.show()

# Create a boxplot of ConvertedSalary
so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()

###What does your data look like? (II)
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Plot pairwise relationships
sns.pairplot(so_numeric_df)

# Show plot
plt.show()

# Print summary statistics
print(so_numeric_df.describe())


'''
Scaling and transformation 
Most common approches standardization and Min-Max scaling 
(sometimes referred to as normalization), and standardization.

Min-Max scaling is when your data is scaled linearly between a minimum 
and masimum value often 0 and 1, with 0 corresponding with the lowest value in
the column and 1 with the largest. 

Min-Max scaling in Python 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df[['Age']])
df['normalized_age'] = scaler.transform(df[['Age']])

Standardization in Python 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
scaler.fit(df[['Age']])
df['standardized_col'] = scalter.transform(df[['Age']])

from sklearn.preprocessing import PowerTransformer 
log = PowerTransormer()
log.fit(df[[‘ConvertedSalary’]])
df[‘log_ConvertedSalary’] = log.transform(df[[‘ConvertedSalary’]])

'''

###Normalization
# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler 

# Instantiate MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_MM'] = MM_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_MM', 'Age']].head())

###Standarization 
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Instantiate StandardScaler
SS_scaler = StandardScaler()

# Fit SS_scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_SS'] = SS_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_SS', 'Age']].head())


###Log transformation
# Import PowerTransformer
from sklearn.preprocessing import PowerTransformer

# Instantiate PowerTransformer
pow_trans = PowerTransformer()

# Train the transform on the data
pow_trans.fit(so_numeric_df[['ConvertedSalary']])

# Apply the power transform to the data
so_numeric_df['ConvertedSalary_LG'] = pow_trans.transform(so_numeric_df[['ConvertedSalary']])

# Plot the data before and after the transformation
so_numeric_df[['ConvertedSalary', 'ConvertedSalary_LG']].hist()
plt.show()

'''
Quantile based detection 
For example you could remove the top 5% 
This is acheived by finding the 95th quatile (the point below which 95% of your data resides)
and removing everything above it

This approach is useful if you are concerned that the highest values in your dataset should 
be avoided 

Quantiles in Python
q_cutoff = df['col_name'].quantile(0.95)
mask = df['col_name'] < q_cutoff
trimmed_df = df[mask]

Standard deviation detection in Python 
mean = df['col_name'].mean()
std = df['col_name'].std()
cut_off = std * 3

lower, upper = mean - cut_off, mean + cut_off
new_df = df[(df['col_name'] < upper) & (df['col_name'] > lower)]
'''

###Train and testing transformations (I)
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print(so_test_numeric[['Age', 'Age_ss']].head())

###Train and test transformation (II)
train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[(so_test_numeric['ConvertedSalary'] < train_upper) \
                             & (so_test_numeric['ConvertedSalary'] > train_lower)]

###Percentage based outlier removal 
# Find the 95th quantile
quantile = so_numeric_df['ConvertedSalary'].quantile(0.95)

# Trim the outliers
trimmed_df = so_numeric_df[so_numeric_df['ConvertedSalary'] < quantile]

# The original histogram
so_numeric_df[['ConvertedSalary']].hist()
plt.show()
plt.clf()

# The trimmed histogram
trimmed_df[['ConvertedSalary']].hist()
plt.show()

###Statistical outlier removal
# Find the mean and standard dev
std = so_numeric_df['ConvertedSalary'].std()
mean = so_numeric_df['ConvertedSalary'].mean()

# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off

# Trim the outliers
trimmed_df = so_numeric_df[(so_numeric_df['ConvertedSalary'] < upper) & (so_numeric_df['ConvertedSalary'] > lower)]

# The trimmed box plot
trimmed_df[['ConvertedSalary']].boxplot()
plt.show()




'''
Scaling and transforming new data 
scaler = StandardScaler()
scaler.fit(train[['col']])
train['scaled_col'] = scaler.transform(train[['col']])

# FIT SOM MODEL
# ....

test = pd.read_csv('test_csv')

test['scaled_col'] = scaler.transform(test[['col']])


Training tranformations for reuse
train_mean = train[['col']].mean()
train_std = train[['col']].std()

cut_off = train_std * 3
train_lower = train_mean - cut_off
train_upper = train_mean + cut_off

# Subset train data 

test = pd.read_csv('test_csv')

# Subset test data 
test = test[(test[['col']] < train_upper) & (test[['col']] > train_lower)]

Why only use training data? 

Data leakage: Using data that you won't have access to when
assessing the performance of your model 

Calibrate your preprocessing steps only on your training data or else you will
overestimate the accuracy of your models 
'''
 

# Print the RawSalary column
print(so_survey_df['RawSalary'])

# Bin the continuous variable ConvertedSalary into 5 bins
so_survey_df['equal_binned'] = pd.cut(so_survey_df['ConvertedSalary'], bins=5)

# Print the first 5 rows of the equal_binned column
print(so_survey_df[['equal_binned', 'ConvertedSalary']].head())


'''
Intriduction to Text Encoding 
Removing unwanted characters 
* [a-zA-Z]: All leter characters 
* [^a-zA-Z]: All non letter characters 

speech_df['text'] = speech_df['text'].str.replace('[^a-zA-Z', '') # the carrot ^ negates everything that is in the braket 

Standardize the case 
speech_df['text'] = speech_df['text'].str.lower()
print(speech_df['text'][0])

Length of text
speech_df['char_cnt'] = speech_df['text'].str.len()
print(speech_df['char_cnt'].head())

speech_df['wrd_cnt'] = 
    speech_df['text'].str.split()
    speech_df['word_cnt'].head(1)
    
word counts
speech_df['word_counts'] = 
    speech_df['text'].str.split().str.len()
print(speech_df['word_splits'].head())

Average length of word
speech_df['avg_word_len'] = speech_df['char_cnt'] / speech_df['word_cnt']
'''

###Cleaning up your text 
# Replace all non letter characters with a whitespace
speech_df['text_clean'] = speech_df['text'].str.replace('[^a-zA-Z]', ' ')

# Change to lower case
speech_df['text_clean'] = speech_df['text_clean'].str.lower()

# Print the first 5 rows of the text_clean column
print(speech_df['text_clean'].head())


###High level text features 
# Find the length of each text
speech_df['char_cnt'] = speech_df['text_clean'].str.len()

# Count the number of words in each text
speech_df['word_cnt'] = speech_df['text_clean'].str.split().str.len()

# Find the average length of word
speech_df['avg_word_length'] = speech_df['char_cnt'] / speech_df['word_cnt']

# Print the first 5 rows of these columns
print(speech_df[['text_clean', 'char_cnt', 'word_cnt', 'avg_word_length']])

'''
Word Count Representation 
Initializing the vectorizor 
from sklearn.feature_extraction.text import CountVectorizer 
cv = CountVectorizer()
print(cv)

Specifying the vectorizer 
from sklearn.eature_extracion.text import CountVectorizer

cv = CountVectorizer(min_df=0.1, max_df=0.9)

min_df: minimum fraction of documents the word must occur 
in max_df: maximum fracion of documents the word can occur in 

cv.fit(speech_df['text_clean'])

cv_transformed = cv.transform(speech_df['text_clean'])
print(cv_transformed)

cv_transformed.toarray()

feature_names = cv.get_feature_names()
print(feature_names)

cv_transformed = cv.fit_transform(speech_df['text_clean'])
print(cv_transformed)

cv_df = pd.DataFrame(cv_transformed.toarray(),
                    columns=cv.get_feature_names()).add_prefix('Counts_')
print(cv_df.head())

Updating your DataFrame
speech_df = pd.concat([speech_df, cv_df, axis=1, sort=False])
print(speech_df.shape)
'''

###Counting words (1)
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate CountVectorizer
cv = CountVectorizer()

# Fit the vectorizer
cv.fit(speech_df['text_clean'])

# Print feature names
print(cv.get_feature_names())


###Counting words (II)
# Apply the vectorizer
cv_transformed = cv.transform(speech_df['text_clean'])

# Print the full array
cv_array = cv_transformed.toarray()
print(cv_array)

# Print the shape of cv_array
print(cv_array.shape)


###Limiting your features
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Specify arguements to limit the number of features generated
cv = CountVectorizer(min_df=0.2, max_df=0.8)

# Fit, transform, and convert into array
cv_transformed = cv.fit_transform(speech_df['text_clean'])
cv_array = cv_transformed.toarray()

# Print the array shape
print(cv_array.shape)


###Text to DataFrame 
# Create a DataFrame with these features
cv_df = pd.DataFrame(cv_array, 
                     columns=cv.get_feature_names()).add_prefix('Counts_')

# Add the new columns to the original DataFrame
speech_df_new = pd.concat([speech_df, cv_df], axis=1, sort=False)
print(speech_df_new.head())


'''
Tf-ldf Representation 

Introducing TF-IDF
print(speech_df['Counts_the'].head())

TF-IDF = (count of word occurances/Total words in document)/log(Number of docs word is in/Total number of docs)
Importing the vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
print(tv)

Max features and stopwords 
tv = TfidfVectorizer(max_features=100, stop_words='english')

max_features : Maximum number of columns created from TF-IDF

stop_words : List of common words to omit e.g. "and", "the" etc. 

tv.fit(train_speech_df['text'])
train_tv_transformed = tv.transformed(train_speech_df['text'])

Putting it all together 
train_tv_df = pd.DataFrame(train_tv_transformed.to_array(),
                            columns=tv.get_feature_names()).add_prefix('TFIDF_')

train_speech_df = pd.concat([train_speech_df, train_tv_df], axis=1, sort=False)

Inspecting your transforms 
examine_rows = train_tv_df.iloc[0]

print(examine_row.sort_values(ascending=False))

Applying the vectorizer to new data 
test_tv_transformed = tv.transform(test_df['text_clean'])

test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), columns=tv.get_feature_name()).add_prefix('TFIDF_')

test_speech_df = pd.concat([test_speech_df, test_tv_df], axis=1, sort=False)
'''

###Tf-idf
# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer  

# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(speech_df['text_clean'])

# Create a DataFrame with these features
tv_df = pd.DataFrame(tv_transformed.toarray(), 
                     columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(tv_df.head())


###Inspecting Tf-idf values
# Isolate the row to be examined
sample_row = tv_df.iloc[0,:]

# Print the top 5 words of the sorted output
print(sample_row.sort_values(ascending=False).head(5))


###Transforming unseen data
# Instantiate TfidfVectorizer
tv = TfidfVectorizer(max_features=100, stop_words='english')

# Fit the vectroizer and transform the data
tv_transformed = tv.fit_transform(train_speech_df['text_clean'])

# Transform test data
test_tv_transformed = tv.transform(test_speech_df['text_clean'])

# Create new features for the test set
test_tv_df = pd.DataFrame(test_tv_transformed.toarray(), 
                          columns=tv.get_feature_names()).add_prefix('TFIDF_')
print(test_tv_df.head())


'''
Bag of words and N-grams 
Issues with bag of words 

Positive meaning 
Single word: happy

Negative meaning 
Bi-gram: not happy

Positive meaning 
Trigram: never not happy

Using N-grams 
tv_bi_gram_vec = TfidfVectorizer(ngram_range = (2,2))

Finding common words 
# Create a DataFrame with the Counts features 
tv_df = pd.DataFrame(tv_bigram.toarray(),
                        columns=tv_bi_gram_vec.get_feature_names().add_prefix('Counts_'))

tv_sums = tv_df.sum()
print(tv_sums.head())

Finding common words 
print(tv_sums.sort_values(ascending=False)).head()
'''


#Fit and apply bigram 
vectorizertv_bi_gram = tv_bi_gram_vec.fit_transform(speech_df['text'])

#Print the bigram features 
print(tv_bi_gram_vec.get_feature_names())

###Using longer n-grams
# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate a trigram vectorizer
cv_trigram_vec = CountVectorizer(max_features=100, 
                                 stop_words='english', 
                                 ngram_range=(3, 3))

# Fit and apply trigram vectorizer
cv_trigram = cv_trigram_vec.fit_transform(speech_df['text_clean'])

# Print the trigram features
print(cv_trigram_vec.get_feature_names())

###Finding the most common words 
# Create a DataFrame of the features
cv_tri_df = pd.DataFrame(data=cv_trigram.toarray(), 
                 columns=cv_trigram_vec.get_feature_names()).add_prefix('Counts_')

# Print the top 5 words in the sorted output
print(cv_tri_df.sum().sort_values(ascending=False).head())



