'''
Machine Learning?
The art and science of:
-Giving computers the ability to learn to make decisions from data 
-without being exlicitly prgrammed!
Examples: 
-Learning to predict whether an email is spam or not 
-Clustering ikipedia entries into different catagories 

*Supervised learning: uses labeled data 
*Unsupervised learning: Uses unlabeled data 

Unsupervised learning: uncovering hidden patterns from unlabeled data 
-Example:
--Grouping customers into distinct categories (Clustering)

Reinforcement learning 
Software agenst inteact with an environment 
-Learn how to optimize their behavior 
-Given a system of rewards and punishments 
-Draws inspiration from behavioral psychology 

Applications: economics, genetics, game playing 

Supervised learning 
*Predictor variables/features and a target variable 
*Aim: Predic the target variable, given the predictor variables 
--Classification: Target variable consists of categories 
--Regression: Target variable is continuous

Naming conventions
*Features = predictor variables = independent variables 
*Target variable = dependent variable = response variable 

Supervised learning 
-Automate time-consuming or expensive manual tasks 
--Example: Doctor's diagnosis 
-Make predictions about the future 
--Example: Will a customer click on an ad or not? 
-Need labeled data
--Historical data with labels 
-Experiemnts to get labeled data 
-Crowd-sourcing labeled data 

Supervised learning in Python 
-We will use scikit-learn/sklearn
--Integrates well with the SciPy stack 
-Other libraries 
--TensorFlow
--keras 

Building a classifier 

k-Nearest Neighbors
-Basic idea: Predict the label of a data point by 
--Looking at the 'k' closest labeled data points 
--Taking a majority vote 

Scikit-learn fit and preict 
*All machine learning models implemented as Python classes 
--They implement the algorithms for learning and predicting 
--Store the information learned from the data 
*Training a model on the data = 'fitting' a model to the data 
-.fit() method
*To predict the labels of new data: .predict() method

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(iris['data'], iris['target']) #pass fit() features: np.array, target: np.array

X-new = np.aray([[5.6, 2.8, 3.9, 1.1],
                [5.7, 2.6, 3.8, 1.3],
                [4.7, 3.2, 1.3, 0.2]])

prediction = knn.predict(X_new)
'''

###k-Nearest Neighbors: Fit
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

###k-Nearest Neighbors: Predict
# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier 

# Create arrays for the features and the response variable
y = df['party'].values
X = df.drop('party', axis=1).values

# Create a k-NN classifier with 6 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the data
knn.fit(X, y)

# Predict the labels for the training data X
y_pred = knn.predict(X)

# Predict and print the label for the new data point X_new
new_prediction = knn.predict(X_new)
print("Prediction: {}".format(new_prediction))


'''
Measuring model performance 
-In classification, accuracy is a commonly used metric 
-Accuracy=Fraction of correct predictions
-Which data should be used to compute accuracy?
How well will the model perform on new data?

Measuring model performance 
-Could compute accuracy on data used to fit classifier 
-NOT indicative of ability to generalize 
-Split data into training and test set 
-Fit/train the classifier on the training set 
-Make predictions on test set 
-Compare predictions with the known labels 

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)
knn = KNeighborsClassifier(n-neighbors=8)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(\"Test set predictions:\\n {}\".format(y_pred))

knn.score(X_test, y_test) #gives the accuracy of our model

Model complexity 
*Larger k = smooother decision boundary = less complex model 
*Smaller k = more complex model = can lead overfitting 

if you increase k even more and less simpler then the model will perform less well on both the training and 
test data 
'''

###The digits recognition dataset 
# Import necessary modules
from sklearn import datasets
import matplotlib.pyplot as plt

# Load the digits dataset: digits
digits = datasets.load_digits()

# Print the keys and DESCR of the dataset
print(digits.keys())
print(digits.DESCR)

# Print the shape of the images and data keys
print(digits.images.shape)
print(digits.data.shape)

# Display digit 1010
plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()


###Train/Test Split + Fit/Predict/Accuracy
# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Create feature and target arrays
X = digits.data
y = digits.target

# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors=7)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))


###Overfitting and Underfitting 
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

'''
Introduction to regression 
X = boston.drop('MEDV', axis=1).values #target is droped 
y = boston['MEDV'].values #keep only the target 

X_rooms = X[:,5]

y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1, 1) #keep the first dimension but add another dimension 1 to xlabel

plt.scatter(X_rooms, y)
plt.ylabel('Value of house /1000 ($)')
plt.xlabel('Number of rooms')
plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression 
reg = LinearRegression()
reg.fit(X_room, y)
prediction_space = np.linpace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()
'''
###Importing data for Supervised learning
# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df.life
X = df.fertility

# Print the dimensions of y and X before reshaping
print("Dimensions of y before reshaping: ", y.shape)
print("Dimensions of X before reshaping: ", X.shape)

# Reshape X and y
y_reshaped = y.reshape(-1,1)
X_reshaped = X.reshape(-1,1)

# Print the dimensions of y_reshaped and X_reshaped
print("Dimensions of y after reshaping: ", y_reshaped.shape)
print("Dimensions of X after reshaping: ", X_reshaped.shape)

'''
Regression mechanics
*y = ax + basestring
-y = target
-x = single features 
-a,b = parameters of model 

*How do we choose a and b? 
*Define an error functions for any given line 
-Choose the line that minimizes the error function 


Linear regression in higher dimensions 
    
        y = a1x1 + a2x2 + b
        
-To fit a linear regression model here: 
--Need to specify 3 variables 

-In higher dimensions: 
--Must specify coefficient for each feature and the variable b 
    
        y = a1x1 + a2x2 + a3x3 + ... + anxn + b

-Scikit-learn API works exactly the same way:
--Pass two arrays: Features, and target 

Linear regression on all features 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

x_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)
reg_all.score(X_test, y_test)
'''


###Fit & predict for regression 
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

###Train test split for regression 
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
  
'''
Cross-validation motivation 
R-squared(model performance) is dependent on the way that the data is split up 
Not representative of the model's ability to generalize 
Solution: Cross-validation!

Cross-validation and model performance 

-5 folds = 5-fold CV
-10 folds = 10-fold CV
-k folds = k-fold CV
More folds = more computationally expensive 

from sklearn.model_selection import cross_val_score 
from sklearn.linear_model import LinearRegression 
reg = LinearRegression
cv_results = cross_val_score(reg, X, y, cv=5) #regressor, feature data, target data, and the number of folds

'''

###5-fold cross-validation
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score 

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
 
 

###K-fold comparison  
 # Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))

'''
Why regularize 
-Recall: Linear regression minimize a loss function
-It chooses a coefficient for each feature variable 
-Large coeffcients can lead to overfitting 
-Penalizing large coeffiecients: Regulalrization

Ridge regression 
-Loss function = OLS loss function + square coefficient for each function * alpha 
Alpha: parameter we need to choose 
Picking alpha here is similar to picking k in k-NN
Hyperparameter tuning (More in Chapter 3)
Alpha controls model complexity 
-Alpha = 0; we get back OLS (Can lead to overfitting)
A very high alpha means that large coefficients are significantly penalized 
This leads to a model that is too simple and ends up underfitting the data 

from sklearn.linear_model import Ridge 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
ridge =Ridge(alpha=0.1, normalize=True) #normalize all variables are on the same scale 
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)

Lasso regression *Loss function = OLS loss function + absolute value of each coefficient * alpha

fromsklearn.linear_model import Lasso 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
lasso = lasso(alpha=0.1, normalize=True)
lasso.fit(X_train,y_train)
lasso_pred = lasso.predict(X_test)
lasso.score(X_test, y_test)

Lasso regression for feature selection
-Can be used to select important features of a dataset 
-Shrinks the coefficients of less important features to exactly 0

from sklearn.linear_model import Lasso 
banes = bonston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()
'''

###Regularization I: Lasso
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.fit(X, y).coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

###Regularization II: Ridge 
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)


'''
How good is your model?
Classification metrics 
-Measuring model performance wih accuracy:
-Fraction of correctly classified samples 
-Not always a useful metric 

Class imbalance exampe: Emails 
-Spam classification--99% of emails are real; 1% of emails are spam 
Could build a classifier that predicts ALL emails as real
-99% accurate!
-But horrible at actually classifying spam 
-Fails at its original purpose 
Need more nuanced metrics

Diagnosing classification predictions 
-Confusion matrix 

|true positives | false negative |
|false positive | true negative  |


*Accuracy: the sum of the diagonal divided by the total sum of the matrix
(tp + tn) / (tp + tn + fp + fn)

*Precision: the numer of true positives divided y the total number of true poositives 
and false positives 
tp / (tp + fp)

*Recall: the number of true positives divided by the total number of true positives 
and false negatives 
tp / (tp + fn) aka sensitivity, hit rate 

*F1score: two times the product of the precision and recall divided by the sum of precision
and recall, the harmonic mean of precision and recall. 
2 * ((precision * recall) / (precision + recall))

 High precision means that the classifier has a low positive rate, that is not many true 
 positives were predicted as beng ture negatives
 
 High recall: predicted most true positives correctly
 
 from sklearn.metrics import classification_report 
 from sklearn.metrics import confusion_matrix
 knn = KNeighborsClassifier(n_neighbors=8)
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
 knn.fit(X_train, y_train)
 y_pred = knn.predict(X_test)
 print(confusion_matrix(y_test, y_pred))
 print(classification_report(y_test, y_pred))
 '''
 
 ###Metrics for classification
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
 
 '''
 Logistic regression and the ROC curve
 
 Logistic regression is used in classification problems not regression problems 
 How does logistic regression work for binary classification?
 -When we have two possible labels for the target variable 
 
 Logistic regression outputs probabilities 
 If the probability 'p' is greater than 0.5
 -The data is labeled '1'
 
 If the probability 'p' is less than 0.5 
 -The data is labeled '0'
 
 from sklearn.linear_model import LogisticRegression
 from sklearn.model_selection import train_test_split 
 logreg = LogisticRegression()
 
 X_train, X_test, y_train, y_test = train_test_split(X, y, rest_size=0.4, random_state=42)
 logreg.fit(X_train, y_train)
 y_pred = logreg.predict(X_test)
 
 The ROC curve 
 The set of points we get when trying all possible thresholds is called the receiver operating 
 characteristics curve 
 or ROC 
 
 plotting the ROC curve 
 from sklearn.metrics import roc_curve
 y_pred_prob = logreg.predict_proba(X_test)[:,1]
 fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) #first argument is given by the actual labels the second by the predictive probabilities
 #fpr: false postive rate 
 #tpr: true positive rate 
 
 _ = plt.plot([0,1], [0,1], 'k--')
 _ = plt.plot(fpr, tpr, label='Logistic Regression')
 _ = plt.xlabel('False Positive Rate')
 _ = plt.ylabel('True Positive Rate')
 _ = plt.title('Logistic Regression ROC Curve')
 plt.show()
 
 We plot these because we do not really want the predictions on our test set but we want the 
 probability that our log reg model outputs before using a threshold to predict the label
 
 logreg.predict_proba(X_test)[:,1]
 '''
 
###Building a logistic regression model 
 # Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


###Plotting the ROC curve 
# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

'''
Area under the ROC curve (AUC)
-The larger the area under the ROC curve = better model

from skleaarn.metrics import roc_auc_score
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred_prob)

AUC can be computed using cross validation 

from sklearn.model_selection import cross_cal_score
cv_scores = cross_val_score(logreg, X, y cv=5, scoring='roc_auc')
print(cv_scores)
'''

###AUC computation
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


'''
Hyperparameter tuning 
-Linear regression: Choosing parameters 
-Ridge/lasso regression: Choosing alpha 
k-Nearest Neighbors: Choosing n_neighbors
Parameters like alpha and k: Hyperparameters 
Hyperparameters cannot be learned by fitting the model 

Choosing the correct hyperparameter 
*Try a bunch of different hyperparameter values 
*Fit all of them separately 
*See how wwell each performs 
*Choose the best performing one 
*It is essential to use cross-validation 


Grid search cross-validation 

from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': np.arrange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.best_params_

knn_cv.best_score_

'''

###Hyperparameter tuning with GridSearch 
# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


###Hyperparameter tuning with RandomizedSearchCV
# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


'''
Hold-out set for final evaluation 
How well can the model perform on never before seen data? 
Using ALL data for cross-validation is not ideal 
Split data into training and hold-out set at the beginning 
Perform grid search cross-validation on training set 
Choose best hyperparameters and evaluate on hold-out set 
'''

###Hold-out set in practice I: Classification 
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)


# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


###Hold-out set in practice II: Regression 
'''
Lasso used the L1 to regulalrize and ridge used the L2 penalty to regulalrize
ridge used the L2 penalty 
There is another type of regulalrized regression known as tthe elastic net. 
In elastic net regulalrization, the penalty term is a linear combination of 
the L1 and L2 penalties: 
        
            a * L1 + b * L2
'''

###Hold-out set in practice II: Regression 
# Import necessary modules
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)


# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)  
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))


'''
Preprocessing data 
Dealing with categorical featues 
Scikit-learn will not accept categorical features by default 
Need to encode categorical features numerically
Convert to dummy variables 
    0: Observation was NOT that category 
    1: Observationo was that category 
    
Some models do not deal with duplicate information well

Dealing with categorical features in Python 
scikit-learn: OneHotEncoder()
pandas: get_dummies()

Encoding dummy variables 
import pandas as pd
df = pd.read_csv('auto.csv')
df_origin = pd.get_dummies(df)
print(df_origin.head())

Linear regression with dummy variables 
from sklearn.model_selection import train_test_split
from sklearn.linear_odel import Ridge 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
ridge = Ridge(alpha=0.5, normalize_state=42)
ridge = Ridge(alpha = 0.5, normalize=True).fit(X_train, y_train)
ridge.score(X_test, y_test)
'''

###Exploring categorical features 
# Import pandas
import pandas as pd

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()


###Creating dummy variables 
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)


###Regression with categorical features 
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)


'''
Handling missing data 
df.insulin.replace(0, np.nan, inplace=True)
df.Tricep.replace(0, np.nan, inplace=True)
df.bmi.replace(0, np.nan, inplace=True)
df.info()

Dropping missing data 
df = df.dropna()
df.shape

Imputing missing data 
-Making an educated guess about the missing values 
-Example: Using the mean of the non-missing entries 

from sklearn.preprocessing import Imputer 
imp = Imputer(missing_values = 'NaN', strategy='mean', axis=0) #axis 0 is for columns
imp.fix(X)
X = imp.transform(X)


Imputing within a pipline
from sklearn.pipline import Pipeline
fom sklean.preprocessing import Imputer 
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
logreg = LogisticRegression()
steps = [('imputation', imp), ('logistic_regression', logreg)]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)pipeline.score(X_test, y_test)
'''

###Dropping missing data 
# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


###Imputing missing data in a ML Pipeline I
# Convert '?' to NaN
df[df == '?'] = np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))


###Imputing missing data in a ML Pipeline II
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))


'''
Centering and scaling
print(df.describe())

Why scale your data?
-Many models use some form of distance to inform them 
-Features on larger scales can unduly influence the model 
-Example: k-NN uses distance explicitly when making predictions 
-We want features to be on a similar scale
-Normalizing (or scaling and centering)

Ways to normalize you data 
-Standardization: Subtract the mean and divide by variance 
-All features are centered around zero and have variance one
-Can also subtract the minimum and divide by the range 
-Minimum zero and maximum one
-Can also normalize so the data ranges from -1 to +1
See scitkit-learn docs for further details 


from sklearn.preprocessing import scale 
X_scaled = scale(X)

np.mean(X), np.std(X)

np.mean(X_scaled), np.std(X_scaled)

Scaling in a pipeline 
from sklearn.preprocessing import StandardScaler 
steps = [('scaler', StandardScaler()), ('knn', KNeighborsClassifier())]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
knn_scaled = pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
accuracy_score(y_test, y_pred)

knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
knn_unscaled.score(X_test, y_test)

steps = [('scaler', StandardScalar()), (('knn', KNeighborsClassifier()))]
pipeline = Pipeline(steps)
parameters = {knn__n_neighbors: np.arange(1,50)}
x_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
cv = GridsearchCV(pipeline, param_grid=parameters)
cv.fit(X_train, y_train)
y_pred = cv.predict(X_test)

print(cv.best_params_)
print(cv.score(X_test, y_test))
print(classification_report(y_test, y_pred))
'''

###Centering and scaling your data 
# Import scale
from sklearn.preprocessing import scale 

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))


###Centering and scaling in a pippeline
# Import the necessary modules
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))


###Bringing it all together I: Pipeline for classification 
# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


###Bringing it all together II: Pipeline for classification
# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
