'''Imputation Techniques 
Basic imputation techniques 
* constant (e.g. 0)
* mean
* median
* mode or most frequent 

Mean Imputation 
from sklearn.impute import SimpleImputer 
diabetes_mean = diabetes.copy(deep=True)
mean_imputer = SimpleImputer(strategy='mean')
diabetes_mean.iloc[:,:] = mean_imputer.fit_transform(diabetes_mean)

diabetes_median = diabetes.copy(deep=True)
median_imputer = SimpleImputer(strategy='median')
diabetes_median.iloc[:,:] = median_imputer.fit_transform(diabetes_median)

Mode imputation 
diabetes_mode = diabetes.copy(deep=True)
mode_imputer = SimpleImputer(strategy='most_frequent')
diabetes_mode.iloc[:,:] = mode_imputer.fit_transform(diabetes_mode)

Imputing a constant 
diabetes_constant = diabetes.copy(deep=True)
constant_imputer = SimpleImputer(strategy='constant', fill_value=0)
diabetes_constant.iloc[:,:] = constant_imputer.fit_transform(diabetes_constant)

Scatterplot of imputation 
diabetes_mean.plot(x='Serum_Insulin', y='Glucose', kind='scatter', alpha=0.5,
                    c=nullity, cmap='rainbow', title='Mean Imputation')

Visualizing imputations 
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
nullity = diabetes['Serum_Insulin'].isnull()+diabetes['Glucose'].isnull()
imputations = {'Mean Imputation': diabetes_mean,
                'Median Imputation': diabetes_median,
                'Most Frequent Impputation': diabetes_mode,
                'Constant Imputation': diabetes_constant}
for ax, df_key in zip(axes.flatten(), imputations):
    imputations[df_key].plot(x='Serum_Insulin', y='Glucose', kind='scatter', alpha=0.5, 
                            c=nullity, cmap='rainbow', ax=ax, colorbar=False, title=df_key)


Summary
You learned to 

* Impute with statistical parameters like mean, median and mode 
* Graphically compare the imputations 
* Analyze the imputations 
'''

###Mean & median imoutation
# Make a copy of diabetes
diabetes_mean = diabetes.copy(deep=True)

# Create mean imputer object
mean_imputer = SimpleImputer(strategy='mean')

# Impute mean values in the DataFrame diabetes_mean
diabetes_mean.iloc[:, :] = mean_imputer.fit_transform(diabetes_mean)

# Make a copy of diabetes
diabetes_median = diabetes.copy(deep=True)

# Create median imputer object
median_imputer = SimpleImputer(strategy='median')

# Impute median values in the DataFrame diabetes_median
diabetes_median.iloc[:, :] = median_imputer.fit_transform(diabetes_median)

###Mode and constant imputation 
# Make a copy of diabetes
diabetes_mode = diabetes.copy(deep=True)

# Create mode imputer object
mode_imputer = SimpleImputer(strategy='most_frequent')

# Impute using most frequent value in the DataFrame mode_imputer
diabetes_mode.iloc[:, :] = mode_imputer.fit_transform(diabetes_mode)

# Make a copy of diabetes
diabetes_constant = diabetes.copy(deep=True)

# Create median imputer object
constant_imputer = SimpleImputer(strategy='constant', fill_value=0)

# Impute missing values to 0 in diabetes_constant
diabetes_constant.iloc[:, :] = constant_imputer.fit_transform(diabetes_constant)



###Visualize imputations
# Set nrows and ncols to 2
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
nullity = diabetes.Serum_Insulin.isnull()+diabetes.Glucose.isnull()

# Create a dictionary of imputations
imputations = {'Mean Imputation': diabetes_mean, 'Median Imputation': diabetes_median, 
               'Most Frequent Imputation': diabetes_mode, 'Constant Imputation': diabetes_constant}

# Loop over flattened axes and imputations
for ax, df_key in zip(axes.flatten(), imputations):
    # Select and also set the title for a DataFrame
    imputations[df_key].plot(x='Serum_Insulin', y='Glucose', kind='scatter', 
                          alpha=0.5, c=nullity, cmap='rainbow', ax=ax, 
                          colorbar=False, title=df_key)
plt.show()

'''
Airquality Dataset 
import pandas as pd
airquality = pd.read_csv('air-quality.csv', parse_dates='Date', index_col='Date')

The .fillna() method
The attribute method in .fillna() can be set to 
* 'ffill' or 'pad'
* 'bfill' or 'backwardfill'


Ffill method
* Replace NaNs with last observable value 
* pad is the same as 'ffill'
airquality.fillna(method='ffill', inplace=True)
airquality['Ozone'][30:40]

airquality.fillna(method='ffill', inplace=True)
airquality['Ozone'][30:40]

Bfill method
* Replace NaNs with next observed value 
* backfill is the same as 'bfill'
df.fillna(method='bfill', inplace=True)

The .interpolate() method 
* The .interpolate() method extends the sequence of values to the missing values 
The attribute method in .interpolate() can be set to 
* 'linear'
* 'quadratic'
* 'nearest'

Linear interpolation
* Impute linearly or with equidistance values 
df.interpolate(method='linear', inplace=True)

airquality['Ozone'][30:40]
airquality.interpolate(method='linear', inplace=True)

Quadratic interpolation 
* Impute the values quadratically 
df.interpolate(method='quadratic', inplace=True)

airquality['Ozone'][30:39]
airquality.interpolate(method='quadratic', inplace=True)

Nearest value imputation
* Impute with the nearest observable value 
df.interpolate(method='nearest', inplace=True)

airquality['Ozone'][30:39]
airquality.interpolate(method='nearest', inplace=True)
'''

###Filling missing time-series data 
# Print prior to imputing missing values
print(airquality[30:40])

# Fill NaNs using forward fill
airquality.fillna(method='ffill', inplace=True)

# Print after imputing missing values
print(airquality[30:40])


###Filling missing time-series data 
# Print prior to imputing missing values
print(airquality[30:40])

# Fill NaNs using backward fill
airquality.fillna(method='bfill', inplace=True)

# Print after imputing missing values
print(airquality[30:40])


###Impute with interpolate method
# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs linearly
airquality.interpolate(method='linear', inplace=True)

# Print after interpolation
print(airquality[30:40])


# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs quadratically
airquality.interpolate(method='quadratic', inplace=True)

# Print after interpolation
print(airquality[30:40])


# Print prior to interpolation
print(airquality[30:40])

# Interpolate the NaNs with nearest value
airquality.interpolate(method='nearest', inplace=True)

# Print after interpolation
print(airquality[30:40])


'''
Visualizing time-series imputations 
Air quaality time-series plot
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

Ffill Imputation 
ffill_imp['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30,5))
airquality['Ozone'].plot(title='Ozone', marker='o')

Bfill Imputation
bfill_imp['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))
airquality['Ozone'].plot(title='Ozone', marker='o')

Linear Interpolation 
linear_interp['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))
airquality['Ozone'].plot(title='Ozone', marker='o')

Quadratic Interpolation 
quadratic_interp['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))
airquality['Ozone'].plot(title='Ozone', marker='o')

Nearest Interpolation
nearest_interp['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30,5))
airquality['Ozone'].plot(title='Ozone', marker='o')

A comparison of interpolations 
# Create subplots
fig, axes = plt. subplots(3, 1 figsize=(30, 20))

# Create interpolations dictionary 
interpolations = {'Linear Interpolation': linear_interp,
                    'Quadratic Interpolation': quadratic_interp,
                    'Nearest Interpolation': nearest_interp}

# Visualize each interpolation 
for ax, df_key in zip(axes, interpolations): 
    interpolations[df_key].Ozone.plot(colo='red', marker='o', linestyle='dotted', ax=ax)
    airquality.Ozone.plot(title=df_key + ' - Ozone', marker='o', ax=ax)
'''
    
    
###Visualize forward fill imputation
# Impute airquality DataFrame with ffill method
ffill_imputed = airquality.fillna(method='ffill')

# Plot the imputed DataFrame ffill_imp in red dotted style 
ffill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

plt.show()


# Impute airquality DataFrame with ffill method
ffill_imputed = airquality.fillna(method='ffill')

# Plot the imputed DataFrame ffill_imp in red dotted style 
ffill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

# Plot the airquality DataFrame with title
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

plt.show()


###Visualize backward fill imputation
# Impute airquality DataFrame with bfill method
bfill_imputed = airquality.fillna(method='bfill')

# Plot the imputed DataFrame bfill_imp in red dotted style 
bfill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

plt.show()

# Impute airquality DataFrame with bfill method
bfill_imputed = airquality.fillna(method='bfill')

# Plot the imputed DataFrame bfill_imp in red dotted style 
bfill_imputed['Ozone'].plot(color='red', marker='o', linestyle='dotted', figsize=(30, 5))

# Plot the airquality DataFrame with title
airquality['Ozone'].plot(title='Ozone', marker='o', figsize=(30, 5))

plt.show()


###Plot interpolations
# Set nrows to 3 and ncols to 1
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(30, 20))

# Create a dictionary of interpolations
interpolations = {'Linear Interpolation': linear, 'Quadratic Interpolation': quadratic, 
                  'Nearest Interpolation': nearest}

# Loop over axes and interpolations
for ax, df_key in zip(axes, interpolations):
  # Select and also set the title for a DataFrame
  interpolations[df_key].Ozone.plot(color='red', marker='o', 
                                 linestyle='dotted', ax=ax)
  airquality.Ozone.plot(title=df_key + ' - Ozone', marker='o', ax=ax)
  
plt.show()

'''
Imputing using fancyimpute
fancyimpute package
* Package contains advanced techniques 
* Uses machine learning algorithms to impute missing values 
* Uses other columns to predict the missing values and impute them 

Fancyimpute imputation techniques 
* KNN or K-Nearest Neighbor
* MICE or Multiple Imputations by Chained Equations 

K-Nearest Neighbor Imputations 
* Select K nearest or similar data points using all the non-missing features 
* Take average of the selected data points to fill the missing feature 

K-Nearest Neighbor Imputation
from fancyimpute import KNN
knn_imputer = KNN()
diabetes_knn = diabetes.copy(deep=True)
diabetes_knn.iloc[:,:] = knn_imputer.fit_transform(diabetes_knn)

Multiple Imputations by Chained Equations (MICE)
* Perform mulltiple regression over random sample of the data 
* Take average of the multiple regression values 
* impute the missing feature value for the data point 

from fancyimpute import IterativeImputer 
MICE_imputer = IterativeImputet()
diabetes_MICE = diebetes.copy(deep=True)
diebetes_MICE.iloc[:,:] = MICE_imputer.fit_transform(diabetes_MICE)

Summary
* Using Machine Learning techniiques to impute missing values 
* KNN finds most similar points for imputing 
* MICE performs multiple regression for imputing
* MICE is a very robust model for imputation
'''

###KNN imputation
# Import KNN from fancyimpute
from fancyimpute import KNN

# Copy diabetes to diabetes_knn_imputed
diabetes_knn_imputed = diabetes.copy(deep=True)

# Initialize KNN
knn_imputer = KNN()

# Impute using fit_tranform on diabetes_knn_imputed
diabetes_knn_imputed.iloc[:, :] = knn_imputer.fit_transform(diabetes_knn_imputed)

###MICE imputation
# Import IterativeImputer from fancyimpute
from fancyimpute import IterativeImputer

# Copy diabetes to diabetes_mice_imputed
diabetes_mice_imputed = diabetes.copy(deep=True)

# Initialize IterativeImputer
mice_imputer = IterativeImputer()

# Impute using fit_tranform on diabetes
diabetes_mice_imputed.iloc[:, :] = mice_imputer.fit_transform(diabetes)


'''
Imputing categorical values 
* Most categorical values are strings 
* Cannot perform operations on strings 
* Necessity to convert/encode strings to numeric values and impute 

Imputation techniques 
* Fill with most frequent category 
* impute using statistical models like KNN

Users profile data 
users = pd.read_csv('userprofile.csv')
users.head()

Ordinal Encoding 
from sklearn.preprocesing import OrdinalEncoder

# Create Ordinal Encoder
ambience_ord_enc = OrdinalEncoder()

# Select non-null values in ambience 
ambience = users['ambience']
ambience_not_null = ambience[ambience.notnull()]
reshaped_vals = ambience_not_null.values.reshape(-1, 1)

# Encode the non-null values of ambience 
encoded_vals = ambience_ord_enc.fit_transform(reshaped_vals)

#Replace the ambience column with ordinal values 
users.loc[ambience.notnull(), 'ambience'] = np.squeeze(encoded_vals)

Ordinal Encoding 
# Create dictionary for Ordinal encoders
ordinal_enc_dict = {}

# Loop over columns to encode 
for col_name in users:
    # Create ordinal encoder for the column
    ordinal_enc_dict[col_name] = OrdinalEncoder()
    
    # Select the non-null values in the column
    col = users[col_name]
    col_not_null = col[col.notnull()]
    reshaped_vals = col_not_null.values.reshaped(-1, 1)
    
    # Encode the non-null values of the column 
    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
    
    
users_KNN-imputed = users.copy(deep=True)

# Create MICE imputer 
KNN_imputer = KNN()

users_KNN_imputed.iloc[:,:] = np.round(KNN_imputer.fit_transform(imputed))

for col in imputed:
    reshaped_col = imputed[col].values.reshape(-1,1)
    users_KNN_imputed[col] = ordinal_enc[col].inverse_transform(reshaped_col)
    
Summary
Steps to impute categorical values 
* Convert non-missing categorical columns to ordinal values 
* Impute the missing values in the ordinal DataFrame 
* Convert back from ordinal values to categorical values 
'''

###Ordinal encoding of a categorical column 
# Create Ordinal encoder
ambience_ord_enc = OrdinalEncoder()

# Select non-null values of ambience column in users
ambience = users['ambience']
ambience_not_null = ambience[ambience.notnull()]

# Reshape ambience_not_null to shape (-1, 1)
reshaped_vals = ambience_not_null.values.reshape(-1,1)

# Ordinally encode reshaped_vals
encoded_vals = ambience_ord_enc.fit_transform(reshaped_vals)

# Assign back encoded values to non-null values of ambience in users
users.loc[ambience.notnull(), 'ambience'] = np.squeeze(encoded_vals)


###Ordinal encoding of a DataFrame 
# Create an empty dictionary ordinal_enc_dict
ordinal_enc_dict = {}

for col_name in users:
    # Create Ordinal encoder for col
    ordinal_enc_dict[col_name] = OrdinalEncoder()
    col = users[col_name]
    
    # Select non-null values of col
    col_not_null = col[col.notnull()]
    reshaped_vals = col_not_null.values.reshape(-1, 1)
    encoded_vals = ordinal_enc_dict[col_name].fit_transform(reshaped_vals)
    
    # Store the values to non-null values of the column in users
    users.loc[col.notnull(), col_name] = np.squeeze(encoded_vals)


###KNN imputation of categorical values 
# Create KNN imputer
KNN_imputer = KNN()

# Impute and round the users DataFrame
users.iloc[:, :] = np.round(KNN_imputer.fit_transform(users))

# Loop over the column names in users
for col_name in users:
    
    # Reshape the data
    reshaped = users[col_name].values.reshape(-1, 1)
    
    # Perform inverse transform of the ordinally encoded columns
    users[col_name] = ordinal_enc_dict[col_name].inverse_transform(reshaped)


'''Evaluation of different imputation techniques 
Evaluation techniques 
* Imputations are used to improve model performance 
* Imputation with maximum machine learning performance is selected 
* Density plots explain the distribution in the data 
* A very good metric to check bias in the imputations 

Fit a linear model for statistical summary 
import statsmodels.api as sm
diabetes_cc = diabetes.dropna(how='any')
X = sm.add_constant(diabetes_cc.iloc[:, :-1])
y = diabetes_cc['Class']
lm = sm.OLS(y, X).fit()

print(lm.summary())

R-squared and Coefficients
lm.rsquared_ad #higher the r^2 the better 

lm.params


Fit linear model on different imputed DataFrames
# Mean Imputation
X = sm.add_constant(diabetes_mean_imputed.iloc[:,:-1])
y = diabetes['Class']
lm_mean = sm.OLS(y, X).fit()

# KNN Imputation 
X = sm.add_constant(diabetes_knn_imputed.iloc[:, :-1])
lm_KNN = sm.OLS(y, X).fit()

# MICE Imputation
X = sm.add_constant(diabees_mice_imputed.iloc[:, :-1])
lm_MICE = sm.OLS(y, X).fit()

Comparing R-squared of different imputation 

print(pd.DataFrame({'Complete': lm.rsquared_adj,
                    'Mean Imp.': lm_mean.rsquared_adj,
                    'KNN Imp.': lm_KNN.rsquared_adj,
                    'MICE Imp.': lm_MICE.rsquared_adj},
                    index=['R_squared_adj']))

Comparing coefficients of different imputation 
print(pd.DataFrame({'Complete': lm.params,
                    'Mean Imp.': lm_mean.params,
                    'KNN Imp.': lm_KNN.params,
                    'MICE Imp.': lm_MICE.params}))

Comparing density plots 
diabetes_cc['Skin_Fold'].plot(kind='kde', c='red', linewidth=3)
diabetes_mean_imputed['Skin_Fold'].plot(kind='kde')
diabetes_knn_imputed['Skin_Fold'].plot(kind='kde')
diabetes_mice_imputed['Skin_Fold'].plot(kind='kde')

labels = ['Baseline (Complete Case)', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']
plt.legend(labels)
plt.xlabel('Skin Fold')


Summary
* Applying linear model from the statsmodels package
* Comparing the coefficients and standard errors 
* Comparing density plots 
'''

###Analyze the summary of linear model
# Add constant to X and set X & y values to fit linear model
X = sm.add_constant(diabetes_cc.iloc[:, :-1])
y = diabetes_cc['Class']
lm = sm.OLS(y, X).fit()

# Print summary of lm
print('\nSummary: ', lm.summary())

# Print R squared score of lm
print('\nAdjusted R-squared score: ', lm.rsquared_adj)

# Print the params of lm
print('\nCoefficcients:\n', lm.params)

# Store the Adj. R-squared scores of the linear models
r_squared = pd.DataFrame({'Complete Case': lm.rsquared_adj, 
                          'Mean Imputation': lm_mean.rsquared_adj, 
                          'KNN Imputation': lm_KNN.rsquared_adj, 
                          'MICE Imputation': lm_MICE.rsquared_adj}, 
                         index=['Adj. R-squared'])

print(r_squared)


###Comparing R-squared and coefficients 
# Store the coefficients of the linear models
coeff = pd.DataFrame({'Complete Case': lm.params, 
                      'Mean Imputation': lm_mean.params, 
                      'KNN Imputation': lm_KNN.params, 
                      'MICE Imputation': lm_MICE.params})

print(coeff)

###Comparing R-squared and coefficients 
r_squares = {'Mean Imputation': lm_mean.rsquared_adj, 
             'KNN Imputation': lm_KNN.rsquared_adj, 
             'MICE Imputation': lm_MICE.rsquared_adj}

# Select best R-squared
best_imputation = max(r_squares, key=r_squares.get)

print("The best imputation technique is: ", best_imputation)


###Comparing density plots 
# Plot graphs of imputed DataFrames and the complete case
diabetes_cc['Skin_Fold'].plot(kind='kde', c='red', linewidth=3)
diabetes_mean_imputed['Skin_Fold'].plot(kind='kde')
diabetes_knn_imputed['Skin_Fold'].plot(kind='kde')
diabetes_mice_imputed['Skin_Fold'].plot(kind='kde')

# Create labels for the four DataFrames
labels = ['Baseline (Complete Case)', 'Mean Imputation', 'KNN Imputation', 'MICE Imputation']
plt.legend(labels)

# Set the x-label as Skin Fold
plt.xlabel('Skin Fold')

plt.show()


