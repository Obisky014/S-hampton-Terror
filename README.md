# S-hampton-Terror
A Logistic Regression Model predicting survivors of the Titanic
Titanic Survivorship Prediction

# Overview

This project aims to predict the survivorship of Titanic passengers using a machine learning model. Given passenger details such as age, sex, passenger class, and fare, the model determines whether a passenger would have survived the disaster.

# Dataset

The dataset used comes from the Titanic disaster and includes the following columns:

PassengerId (int) - Unique identifier for each passenger

Survived (int) - Target variable (1 = Survived, 0 = Did not survive)

Pclass (int) - Passenger class (1st, 2nd, 3rd)

Name (object) - Passenger name (dropped in preprocessing)

Sex (object) - Gender of the passenger

Age (float) - Age of the passenger (contains missing values)

SibSp (int) - Number of siblings/spouses aboard

Parch (int) - Number of parents/children aboard

Ticket (object) - Ticket number (dropped in preprocessing)

Fare (float) - Fare paid for the ticket

Cabin (object) - Cabin number (dropped in preprocessing due to high missing values)

Embarked (object) - Port of embarkation (encoded)

# Data Preprocessing

Handling Missing Values:

Age: Imputed using group-wise median based on Pclass and Sex

Cabin: Dropped due to excessive missing values

Embarked: Encoded (one-hot encoding)

Feature Selection:

Dropped PassengerId, Name, Ticket, and Cabin

Encoding Categorical Variables:

Sex encoded as binary (0 for male, 1 for female)

Embarked one-hot encoded

# Splitting the Dataset:

80% training, 20% testing

# Model Selection & Training

The primary model used for classification is Logistic Regression, as this is a binary classification problem. The model was trained on the preprocessed dataset using the scikit-learn library.

# Model Evaluation

The model was evaluated using:

Accuracy: 80%

Confusion Matrix:

True Negatives: 94

False Positives: 14

False Negatives: 21

True Positives: 49

Classification Report:

Precision, Recall, and F1-score for each class

How to Run

Prerequisites

Python 3.x

Pandas

NumPy

Scikit-learn

Matplotlib (for visualizations)

# Further Notes
This was basically my first time understanding confusion matrix and well, talk about its name being on brand.
The confusion matrix essentially breaks down the predictions into four categories:

# True Negatives (TN):
Passengers who did not survive, and the model correctly predicted they did not survive.
# False Positives (FP):
Passengers who did not survive, but the model incorrectly predicted they survived.
# False Negatives (FN):
Passengers who did survive, but the model incorrectly predicted they did not survive.
# True Positives (TP):
Passengers who did survive, and the model correctly predicted they survived.

So, the confusion matrix essentially splits the two groups (survived and not survived) into correct and incorrect predictions:

# The first row [94 14]:

94 are passengers who did not survive, and the model correctly predicted they didn't survive (True Negative).
14 are passengers who did not survive, but the model incorrectly predicted they survived (False Positive).

# The second row [21 49]:

21 are passengers who did survive, but the model incorrectly predicted they did not survive (False Negative).
49 are passengers who did survive, and the model correctly predicted they survived (True Positive).

To put it simply, the matrix shows how well the model has classified each group and whether it made the right prediction (True) or wrong prediction (False). This is why you have two numbers for both the survived and not survived groups—each representing the number of correct vs. incorrect predictions.

You may also wonder,  how does the model know who and who did not survive, considering we dropped the column of survived?
well, When we dropped the Survived column, we only removed it from the features (X) used for training, not from the dataset entirely.
X contains all the independent variables (features) used for prediction.
y contains only the dependent variable (the actual survival labels: 0 = not survived, 1 = survived).
When we train the logistic regression model with;
# model.fit(X_train, y_train)  (i don't know how to make it small and bold at the same time ;))
The model learns patterns in X_train (passenger class, sex, age, fare, etc.).
It maps these patterns to the corresponding values in y_train (whether they survived or not).
Then, when making predictions we use;
# y_pred = model.predict(X_test) (again, i still don't know how to make it tiny and brave at the same time :()
The model looks at the X_test data and predicts whether each passenger survived (1) or not (0).
We then compare these predictions (y_pred) with the actual values (y_test) to evaluate how well the model performed.

We compare the predicted values (y_pred) with the actual survival values (y_test).
The confusion matrix is then created from this comparison.
So even though we dropped Survived from X, we kept it in y, which allows us to compare what the model predicted vs. what actually happened.


