### Classification with XGBoost 
'''
Basics of:
    -Supervised classification
    -Decision trees
    -Boosting

Problems that XGBoost can be applied to, relies on labeled data
Supervised Learning 
* Relies on labeled data 
* Have some understanding of past behavior 

There are two kinds of supervised learning problems that account for the vast 
majority of use cases: classification problems and prediction problems  

Example: 
Does a specific image contain a person's face?
    * Training data: vectors of pixel values
    * Labels: 1 or 0 

Binary classification example 
    -Will a person purchase the insurance package given some quote? 
        binary supervised learning problem 
    -Predicting whether one of several species of birds is a multi-
    class supervised learning problem. 

AUC: Metric for binary classification models, most versatile and 
common evaluation metric used to judge the quality of binary classification model  
    * Area under the ROC curve (AUC)
    * Larger area under the ROC curve = better model

Thep probability that a randomely chosen positive data point will have a higher rank than 
than a randomly chosen negative data point for your learning problem. 
    
    A higher AUC means more sensitive, better performing model 
    
Multi-class classification problems it is common to use the accuracy score
* Confusion matrix 

                    Predicted:              Predicted:
                    Spam Email              Real Email
Actual: Spam Email  True Positive           False Negative 
Actual: Real Email  False Positive          True Negative

Accuracy
                        tp + tn
                   tp + tn + fp + fn

Other supervised learning considerations 
* Features can be either numeric or catagorical
* Numeric features should be scaled (Z-scored)

Recommendation 
* Recommending an item to a user 
* Based on consumption history and profile 
Example: Netflix
'''

'''
Hottest library in supervised learning XGBoost
What is XGBoost?
Optimized gradient-boosting machine learning library 
Originally written in C++
Has APIs in several languages:
-Python
-R
-Scala
-Julia
-Java

What makes XGBoost so popular?
Speed and performance 
Core algorithm is parallelizable 
Consistently outperforms outperforms single-algorithm methods 
State-of-the-art performance in many ML tasks 


###Example: XGBoost in Classification problem 

import xgboost as xgb
import pandas as pd 
import numpyt as np
from sklearn.model_selection import train_test split  
class_data = pd.read_csv("classification_data.csv")

X, y = class_data.iloc[:,:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)
xg_cl.fit(X_train, y_train)

preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]

print("accuracy: %f" % (accuracy))

OUTPUT>> accuracy: 0.78333
'''

 ###XGBoost: Fit/Predict
 # Import xgboost
import xgboost as xgb

# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the training and test sets
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBClassifier: xg_cl
xg_cl = xgb.XGBClassifier(objective='binary:logistic', n_estimators=10, seed=123)

# Fit the classifier to the training set
xg_cl.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_cl.predict(X_test)

# Compute the accuracy: accuracy
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

'''
What is a decision tree? 
*Base learner - individual learning algorithm in an ensemble algorithm 
*Composed of a series of binary questions 
*Predictions happen at the "leaves" of the trees 

Decision tres and CART 
*Constructed iteratively (one decision at a time)
-Until a stopping criterion is met 

CART: Classification and Regression Trees
*Each leaf always contains a real-value score 
*Can later be conveted into categories 
'''

###Decision Trees
# Import the necessary modules
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 

# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the classifier: dt_clf_4
dt_clf_4 = DecisionTreeClassifier(max_depth=4)

# Fit the classifier to the training set
dt_clf_4.fit(X_train, y_train)

# Predict the labels of the test set: y_pred_4
y_pred_4 = dt_clf_4.predict(X_test)

# Compute the accuracy of the predictions: accuracy
accuracy = float(np.sum(y_pred_4==y_test))/y_test.shape[0]
print("accuracy:", accuracy)

'''
Boosting overview 
*Not a specific machine learning algorithm 
*Concept that can be applied to a set of machine learning models 
-Meta-algorithm 
*Ensemble meta-algorithm used to convert many weak learners into a strong learner 

Weak learners and strong leaners 
*Weak learner: ML algorithm that is slightly better than chance 
-Example: Decision tree whose predictions are slightly better than 50%
*Boosting converts a collection of weak learners into a strong learner 
*Strong learner: Any algorithm that can be tuned to achieve good performance 

How boosting is accomplished 
*Iteratively learning a set of weak models on subsets of the data 
*Weighing each weak prediction according to each weak learner's performance 
*Combine the weighted predictions to obtain a single weighted prediction that 
is much better than the individual predictions themselves 

Model evaluation through cross-validation
*Cross-validation: Robust method for estimating the performance of a model 
on unseen data
*Generating many non-overlapping train/test splits on training data 
*Reports the average test set performance across al data splits 

Cross-validation in XGBoost example 
import xgboost as xgb
import pandas as pd
churn_data = pd.read_csv("classification_data.csv")
churn_dmatrix = xgb.DMatrix(data=churn_data.iloc[:,:-1], label=churn_data.month_5_still_here)
params={"objective":"binary:logistic","max_depth":4}
cv_results=xgb.cv(dtrain=chur_dmatrix, params=params, nfold=4, num_boost_round=10, metrics="error", as_pandas=True)
print("Accuracy: %f" %((1-cv_results["test-error-mean"']).iloc[-1]))
'''

###Measuring accuracy 
# Create arrays for the features and the target: X, y
X, y = churn_data.iloc[:,:-1], churn_data.iloc[:,-1]

# Create the DMatrix from X and y: churn_dmatrix
churn_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:logistic", "max_depth":3}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="error", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the accuracy
print(((1-cv_results["test-error-mean"]).iloc[-1]))

###Measuring AUC
# Perform cross_validation: cv_results
cv_results = xgb.cv(dtrain=churn_dmatrix, params=params, 
                  nfold=3, num_boost_round=5, 
                  metrics="auc", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Print the AUC
print((cv_results["test-auc-mean"]).iloc[-1])


'''
When should you use XGBoost? 
*You have a large number of training samples 
-Greater than 1000 training samples and less 100 features 
-The number of features < number of training samples 
*You have a mixture of categorical and numeric features 
-Or just numeric features 

When to NOT use XGBoost 
*Image recognition 
*Computer vision
*Natural language processing and understanding problems
*Whe the number of training samples is significantly smaller 
than the number of features   

Regression basics
*Outcome is real-value 
    ie. predicting the hieght in centimeters, you are solving a regression problem 

Evaluating a regression model uses different kinds of metrics than those that 
we described for use in classification problems in chapter 1, is most cases we use 
root mean squared error (RMSE) or the mean absolute error (MAE) to evaluate the 
quality of a regression model. RMSE is computed by taking the difference between 
the actual and the predicted values for what you are trying to predict, squaring 
the differences, computing their mean, and taking that value's square root and 
taking there values square root. This allows us to treat negative and positive 
differences equally, but tends to punish larger differences between predicted and 
actual values much more than smaller ones. 

MAE simply sums the absolute differences between predicted and actual values 
across all of the samples that we build our model on. Although MAE isn't affected 
by large differences as much as RMSE, it lacks some nice mathemaatical prperties that 
make it much less frequently used as an evaluation metric. Some common algorithms 
that are used for regression problems include linear regression and decision trees

Objective (loss) functions and base learners

Objective Functions and Why We Use Them 
* Quantifies how far off a prediction is from the actual result 
* Measures the difference between estimated and true values for some collection of 
data 

Goal: Find the model that yields the minimum value of the loss function

Common loss functions and XGBoost
*Loss function names in xgboost:
-reg:linear - use for regression problems 
-reg:logistic - use for classification 
problems when you want just decision, not probability
-binary:logistic - use when you want probability rather than just decision 

XGBoost is an ensemble learning method composed of many individual models that 
are added together to generate a single prediction 
Individual ms
models = base learners 
Each base learner should be good at distinguishing or predicting different parts
of the dataset 
base learners that when combined create final prediction that is non-linear

Goal: The goal of XGBoost is to have base learners that is slightly bettr than 
random guessing on certain subsets of training examples, and uniformly bad at 
the remainder, so that when all of the predictions are combined, the uniformly
bad predictions cancel out and those sligtly better than chance combine into a 
single very good prediction 

Two kinds of base learners: tree and linear 

Trees as base learners example: Scikit-learn API 

*********
import xgboost as xgb
import pandas as pd
import nmpy as np
from sklearn. model_selection import train_test_split 

boston_data = pd.read_csv("boston_housing.csv")
X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

xg_reg = xgb.XGBRegressor(objective='reg:linear', n_estimators=10, seed=123)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))

*********
import xgboost as xgb
import pandas as pd
import nmpy as np
from sklearn. model_selection import train_test_split 

boston_data = pd.read_csv("boston_housing.csv")
X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test = xgb.DMatrix(data=X_test, label=y_test)
params = {"booster":"gblinear", "objective":"reg:linear"}
xg_reg = xgb.train(params = params, dtrain=DM_train, num_boost_round=10)

preds = xg_reg.predict(DM_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse))
'''

###Decision trees and base learners 
# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective="reg:linear", n_estimators=10, seed=123)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


###Linear base learners 
# Convert the training and testing sets into DMatrixes: DM_train, DM_test
DM_train = xgb.DMatrix(data=X_train, label=y_train)
DM_test =  xgb.DMatrix(data=X_test, label=y_test)

# Create the parameter dictionary: params
params = {"booster":"gblinear", "objective":"reg:linear"}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=DM_train, num_boost_round=5)

# Predict the labels of the test set: preds
preds = xg_reg.predict(DM_test)

# Compute and print the RMSE
rmse = np.sqrt(mean_squared_error(y_test,preds))
print("RMSE: %f" % (rmse))


###Evaluating model quality 
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-rmse-mean"]).tail(1))

OUTPUT:  train-rmse-mean  train-rmse-std  test-rmse-mean  test-rmse-std
    0    141767.531250      429.448328   142980.433594    1193.791602
    1    102832.542969      322.473304   104891.392578    1223.157623
    2     75872.619140      266.472468    79478.937500    1601.344539
    3     57245.651367      273.625016    62411.921875    2220.149857
    4     44401.298828      316.423666    51348.280274    2963.379319
    4    51348.280274
    
    
###Evaluating model quality 
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=4, num_boost_round=5, metrics="mae", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)

# Extract and print final boosting round metric
print((cv_results["test-mae-mean"]).tail(1))

OUTPUT: train-mae-mean  train-mae-std  test-mae-mean  test-mae-std
    0   127343.480469     668.348102  127633.976562   2404.004666
    1    89770.052735     456.962096   90122.503906   2107.914523
    2    63580.790039     263.404933   64278.561524   1887.563452
    3    45633.153321     151.886440   46819.168945   1459.818607
    4    33587.092774      86.999100   35670.645508   1140.606558
    4    35670.645508

'''
Regularization and base learners in XGBoost 
Loss functions and EXBoos don't just take into account how close a model's 
how close a models predictions are to the actual value but also take into 
account how complex the model is. This isea of penalizing models as they 
become more complex is called regularization. 

Loss functions and XGBoost are used to find models that are both accurate 
and as simple as they can be. There are several parameters that can be 
tweaked in XGBoost to limit model complexity y altering the loss function 
 
Gamma is a parameter for tree base learners that controls whether a given 
node on a base learner will split based on the expected reduction in the 
loss that would occur after performing the split, so that highe values lead 
to fewer splits 

Alpha is another name for L1 regularization, this regularization term is 
a penalty on leaf weights rather than on feature weights, as is the case 
in linear or logistic regression. Higher alpha values lead to stronger L1
regularization, which causes many leaf weights in the base learners to go 
to 0. Lambda is another name for L2 regularization. L2 regularization is 
a much smoother penalty than L1 and causes leaf weights to smoothly 
decrease, instead of enforcing stong sparsity constraints on the leaf 
weights as in L1. 

Regularization in XGBoost 
-Regularization is a control on model complexity 
-Want models that are both accurate and as simple as possible 
-Regularization parameters in XGBoost:
**gamma - minimum loss reduction allowed for a split to occur 
**alpha - l1 regularization on leaf weights, larger values mean more 
regularization 
**lambda - l2 regularization on leaf weights 


Example:
import xgboost as xgb
import pandas as pd 
boston_data = pd.read_csv("boston_data.csv")
X, y = boston_data.iloc[:,:-1], boston_data.iloc[:,-1]
bosto_dmatrix = xgb.DMatrix(data=X, label=y)
params={"objective":"reg:linear","max_depth":4}
l1_params = [1, 10, 100]
rmses_l1=[]

for reg in l1_params:
    params['alpha'] = reg
    cv_results = xgb.cv(dtrain=boston_dmatrix, params, nfold=4, 
        num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)
    rmses_l1.append(cv_results["test-rmse-mean"].tail(1).values[0])
print("Best rmse as a function of l1:")
print(pd.DataFrame(list(zip(l1_params, rmses_l1)), columns=["l1","rmse]))

OUTPUT:
    l1           rmse
 0   1    69572.517742
 1  10    73721.967141
 2 100    82312.312413
 
 
Comparison of Base Learners in XGBoost 
The linear base learner is a sum of linear terms, exactly as you would find in 
linear or logistic regression model. When you combine many of these base 
models into an ensemble, you get a weighted sum of linear models, which is itself
linear. 

Since you don't get any nonlinear combination of features in the final model, this
approach is rarely used, as you can get identical performcances from regularized 
linear model

The tree base learner uses decision trees as base models. When the decision trees 
are all combined into an ensemble, there combination becomes a nonlinear function
of each individual tree, which itself is nonlinear.

* Linear Base Learner 
    -Sum of linear terms 
    -Boosted model is weighted sum of a linear model (thus is itself linear)
    -Rarely used 
* Tree Base Learner:
    -Decision tree
    -Boosted model is weighted sum of decision trees (nonlinear)
    -Almost exclusively used in XGBoost 

The list and zip function will be used to to convert multiple equal-length list into 
a single object that we can convert into a pandas dataframe 

Zip is a function that allows you to take multiple equal-legth lists and iterate 
over them in parallel, side by side, in python3 zip creates a generator , or an 
object that doesn't have to be completely instantiated at runtime. In order for
the entire zipped pair of lists to be instantiated, we have to case the zip 
generator object into a list directly. After casting, we can convert this object 
directly into a dataframe


*pd.DataFrame(list(zip(list1, list2)), columns=["list1", "list2"])

*zip creates a generator of parallel values:
- zip([1,2,3],["a","b","c"]) = [1, "a"],[2,"b"],[3,"c"]

- generators need to be completely instantiated before they can be used in DataFram 
objects 

*list() instantiates the full generator and passing that into the DataFrame 
converts the whole expression  
'''

###Using regularization in XGBoost 
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:linear", "max_depth":3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg
    
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))

OUTPUT: 
    l2          rmse
0    1  52275.359375
1   10  57746.064453
2  100  76624.625000

#As the value of labda increases so does rmse

###Visualizing individual XGBoost trees 
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg, num_trees=0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg, num_trees=4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg, num_trees=9, rankdir="LR")
plt.show()

###Visualizing feature importances: What features are most important in my dataset 
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()


'''
Why tune your model?
Untuned model example 

import pandas as pd 
import xgboost as xgb
import numpy as np

housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X,y = housing_data[houosing_data.columns.tolist()[:-1]], 
    housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
untuned_params = {"objective":"reg:linear"}
untuned_cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, 
    params=untuned_params, nfold=4, metrics="rmse", ,as_pandas=True, seed=123)
print("Untuned rmse: %f" %((untuned_cv_results_rmse["test-rmse-mean"]).tail(1)))

Output:
Untuned rmse: 34624.229988  # $34,624.23


Tuned model example

import pandas as pd 
import xgboost as xgb
import numpy as np

housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X,y = housing_data[houosing_data.columns.tolist()[:-1]], 
    housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
tuned_params = {"objective":"reg:linear",'colsample_bytree': 0.3, 
    'learning_rate':0.1, 'max_depth':5}
tuned_cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, 
    params=tuned_params, nfold=4, num_boost_round=200, metrics="rmse", as_pandas=True, seed=123)
print("Tuned rmse: %f" %((tuned_cv_results_rse["test-rmse-mean"]).tail(1)))

Output:
Tune rmse: 29812.683594  # $29,812.68 14% reduction in RMSE
'''
###Tuning the number of boosting rounds

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params 
params = {"objective":"reg:linear", "max_depth":3}

# Create list of number of boosting rounds
num_rounds = [5, 10, 15]

# Empty list to store final round rmse per XGBoost model
final_rmse_per_round = []

# Iterate over num_rounds and build one model per num_boost_round parameter
for curr_num_rounds in num_rounds:

    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, num_boost_round=curr_num_rounds, metrics="rmse", as_pandas=True, seed=123)
    
    # Append final round RMSE
    final_rmse_per_round.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
num_rounds_rmses = list(zip(num_rounds, final_rmse_per_round))
print(pd.DataFrame(num_rounds_rmses,columns=["num_boosting_rounds","rmse"]))

Output:
       num_boosting_rounds          rmse
    0                    5  50903.298177
    1                   10  34774.192708
    2                   15  32895.098958
    
###Automated boosting round selection using early_stopping

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree: params
params = {"objective":"reg:linear", "max_depth":4}

# Perform cross-validation with early stopping: cv_results
cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, early_stopping_rounds=10, num_boost_round=50, metrics="rmse", as_pandas=True, seed=123)

# Print cv_results
print(cv_results)


'''
Overview of XGBoost's hyperparameters 

The learning rate affects how quickly the model fits the residual error using additional
base learners. A low learning rate will require more boosting rounds to achieve the same 
reduction in residual error as an XGBoost model with a high learning rate. Gamma, alpha, 
and lambda all have an effect on how strongly regularized the trained model will be. 
Max_depth must be a positive intiger vlue and affects how deeply each tree is allowed to 
grow during any given boosting round. Subsample must be a value between 0 and 1 and is 
the fraction of the total training set that can be used for any given boosting round and 
must be a value between 0 and 1. A large value means that almost all features can be used
to build a tree during a given boosting round, whereas a small value means that the fraction 
of features that can be selected from is very small. In general, smaller colsample_bytree 
values can be thought of as providing additional regulalrization to the model, whereas 
using all coloumns may contain certain cases overfit a trained model.             

Common tree tunable parameters 
*Learning rate: learning rate/eta
*gamma: min loss reduction to create new tree split 
*lambda: L2 reg on leaf weights 
*alpha: L1 reg on leaf weights 
*max_depth: max depth per tree 
*subsample: % samples used per tree 
*colsample_bytree: % features used per tree 


Linear tunable parameters 
For linea base learner, the number of tunable parameters is significantly smaller 
You only have access to L1 and L2 regularization on the weights associated with 
any given feature, and then another regularization term that can be applied the 
models bias. Its important to mention that the number of boosting rouds (that is 
either the number of trees you build or the number of linear base learners you 
construct) is itself a tunable parameter 


* lambda: L2 reg on weights 
* alpha: L1 reg on weights 
* lambda_bias: L2 reg term on bias
* You can also tune the number of esimators used for both base model types 
'''
###Tuning eta

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary for each tree (boosting round)
params = {"objective":"reg:linear", "max_depth":3}

# Create list of eta values and empty list to store final round rmse per xgboost model
eta_vals = [0.001, 0.01, 0.1]
best_rmse = []

# Systematically vary the eta 
for curr_val in eta_vals:

    params["eta"] = curr_val
    
    # Perform cross-validation: cv_results
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=3, early_stopping_rounds=5, num_boost_round=10, metrics="rmse", as_pandas=True, seed=123)

    
    
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(eta_vals, best_rmse)), columns=["eta","best_rmse"]))


###Tuning max_depth 

# Create your housing DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params = {"objective":"reg:linear"}

# Create list of max_depth values
max_depths = [2, 5, 10, 20]
best_rmse = []

# Systematically vary the max_depth
for curr_val in max_depths:

    params["max_depth"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(max_depths, best_rmse)),columns=["max_depth","best_rmse"]))

###Tuning cosample_bytree 
# Create your housing DMatrix
housing_dmatrix = xgb.DMatrix(data=X,label=y)

# Create the parameter dictionary
params={"objective":"reg:linear","max_depth":3}

# Create list of hyperparameter values: colsample_bytree_vals
colsample_bytree_vals = [0.1, 0.5, 0.8, 1]
best_rmse = []

# Systematically vary the hyperparameter value 
for curr_val in colsample_bytree_vals:

    params["colsample_bytree"] = curr_val
    
    # Perform cross-validation
    cv_results = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2,
                 num_boost_round=10, early_stopping_rounds=5,
                 metrics="rmse", as_pandas=True, seed=123)
    
    # Append the final round rmse to best_rmse
    best_rmse.append(cv_results["test-rmse-mean"].tail().values[-1])

# Print the resultant DataFrame
print(pd.DataFrame(list(zip(colsample_bytree_vals, best_rmse)), columns=["colsample_bytree","best_rmse"]))

'''
Review of grid search and random search 
How do we find the optimal values for several hyperparameters simultaneously, leading to 
the lowest possible, when their values interact in non-obvious, non-linear way?

Two common strategies for choosing several hyperparameter values simultaneously are 
Grid Search and Random Search, that we review them here, and see what their advantages
and disadvantages are, by looking at some examples of how both can be used with XGBoost 
and scikit-learn packages.

Grid Search is a method of exhaustively searching through a collection of possible parameter
values. For example, if you have 2 hyperparameters you would like to tune, and 4 possible 
values for each hyperparameter, then a grid search over that parameter space would try all 
16 possible parameter configurations. In a grid search, you try every parameter configuration, 
evaluate some metric for that configuration, and pick the parameter configuration that gave you 
the best value for the metric you were using, which in our case will be the root mean squared
error. 

Grid search: review 
* Search exhaustively over a given set of hyperparameters, once per set of hyperparameters 
* Number of models = number of distinct values per hyperparameter multiplied across each 
hyperparameter 
* Pick finall model hyperparameter values that give best cross-validated evaluation 
metric value 

Let's go over an example of how to grid search over several hyperparameters using 
XGBoost and scikit learn. 


import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
housing_data= pd.read_csv("ames_housing_trimmed_processed.csv")
X, y = housing_data[housing_data.columns.tolist()[:-1]],
       housing_data[housing_data.columns.tolist()[-1]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {'learning_rate': [0.01, 0.1, 0.5, 0.9],
                    'n_estimators': [200],
                    'subsample': [0.3, 0.5, 0.9]}
gbm = xgb.XGBRegressor()
grid_mse = GridSearchCV(estimator=gbm, param_grid,
            scoring='neg_mean_squared_error', cv=4, verbose=1)
grid_mse.fit(X, y)
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


Random search is significantly different from grid search in that the number of models 
that you are required to iterate over doesn't grow as you expand the overall 
hyperparameter space. In random search, you get to decisde how many models, or iterations, 
you want to try out before stopping. Random search simply involves drawing a random combination 
of possible hyperparameter values from the range of alowable hyperparameters a set number of 
times. Each time, you train a model with the selected hyperparameters, evaluate the performance
of that model, and then rinse and repeat. When you've created the number of models you had 
specified initially, you simply pick the best one. 

Random search: review 
* Create a (possibly infinite) range of hyperparameter values per hyperparameter that you would 
like to search over 
* Set the number of iterations you would like for the random search to continue 
* During each iteration randomly draw a value in the range of specified values for each
hyperparameter search over and train/evaluate a model with those hyperparameters
* After you've reached the maximum number of iterations, select the hyperparameter configuration 
with the best evaluated score. 

Random Search Example 

import pandas as pd 
import xgboost as xgb
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
housing_data = pd.read_csv("ames_housing_trimmed_processed.csv")
X, y = housing_data[housing_data.columns.tolist()[:-1],
        housing_data[housing_data.columns.tolist()[-1]]
housing_dmatrix = xgb.DMatrix(data=X, label=y)
gbm_param_grid = {'learning_rate': np.range(0.05, 1.05, .05),
                    'n_estimamtors': [200],
                    'subsample': np.arange(0.05, 1.05, .05)}
gbm = xgb.XGBRegressor()
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid,
                                    n_iter=25, scoring='neg_mean_squared_error', cv=4, verbose=1)
randomized_mse.fit(X, y)
randomized_mse.fit(X, y)
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))
'''


###Grid search with XGBoost
# Create the parameter grid: gbm_param_grid
gbm_param_grid = {
    'colsample_bytree': [0.3, 0.7],
    'n_estimators': [50],
    'max_depth': [2, 5]
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor()

# Perform grid search: grid_mse
grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring='neg_mean_squared_error', cv=4, verbose=1)


# Fit grid_mse to the data
grid_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", grid_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))


###Random search with XGBoost
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': [25],
    'max_depth': range(2, 12)
}

# Instantiate the regressor: gbm
gbm = xgb.XGBRegressor(n_estimators=10)

# Perform random search: grid_mse
randomized_mse = RandomizedSearchCV(estimator=gbm, param_distributions=gbm_param_grid, n_iter=5, scoring='neg_mean_squared_error', cv=4, verbose=1)


# Fit randomized_mse to the data
randomized_mse.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", randomized_mse.best_params_)
print("Lowest RMSE found: ", np.sqrt(np.abs(randomized_mse.best_score_)))

'''
Limits of grid search and random search 
Grid search and random search each suffer from distinct limitations 

As long as the number of hyperparameters and distinct values per hyperparameter you search 
over is kept small, grid search will give you an answer in a reasonable amount of time. 
However the hyperparameters grows, the time it takes to completely a full grid search 
increases exponentially

Grid Search 
*Number of models you must build with every additional new parameter grows very quickly


For random search, the problem is a bit different. Since you can specify how many iterations 
a random search shuold be run, the time it takes to finish the random search wont explode 
as you add more and more hyperparameters to search through. Te problem really is that you 
add new hyperparameters to search over, the size of the hyper parameter space explodes as it 
did in the grid search case, and so you are left hoping that one of the random parameter 
configurations that the search chooses is a good one

Random Search
*Parameter space to explor can be massive