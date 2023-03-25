# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 03:39:14 2023

@author: Yunus

MIT Open Learning

Introduction to Data-Centric AI

Label Error

Lab 2

Source:
    https://github.com/dcai-course/dcai-lab/blob/master/label_errors/solutions/Solution%20-%20Label%20Errors.ipynb
"""

""" 
In this lab, we will:

Establish a baseline XGBoost model accuracy on the original data
Automatically find mislabeled data points by:
Computing out-of-sample predicted probabilities
Estimating the number of label errors using confident learning
Ranking errors, using the number of label errors as a cutoff in identifying issues
Remove the bad data
Retrain the exact same XGBoost model to see the improvement in test accuracy

"""

# pip install xgboost==1.7 scikit-learn pandas cleanlab

from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

df = pd.read_csv("student-grades.csv")
print(df.head())

df_c = df.copy()
# Transform letter grades and notes to categorical numbers.
# Necessary for XGBoost.
df['letter_grade'] = preprocessing.LabelEncoder().fit_transform(df['letter_grade'])
df['noisy_letter_grade'] = preprocessing.LabelEncoder().fit_transform(df['noisy_letter_grade'])
df['notes'] = preprocessing.LabelEncoder().fit_transform(df["notes"])
df['notes'] = df['notes'].astype('category')

print(df.head())

# To apply confident learning, we need to obtain out-of-sample predicted probabilities for all of our data. 
# To do this, we can use K-fold cross validation: for each fold, we will train on some subset of our data and get predictions on the rest of the data that was not used for training.

# Prepare training data (remove labels from the dataframe) and labels
data = df.drop(['stud_ID', 'letter_grade', 'noisy_letter_grade'], axis=1)
labels = df['noisy_letter_grade']

# XGBoost(experimental) supports categorical data.
# Here we use default hyperparameters for simplicity.
# Get out-of-sample predicted probabilities and check model accuracy.
model = XGBClassifier(tree_method="hist", enable_categorical=True)

# Compute out-of-sample predicted probabilities for every data point. 
# We can do this manually using for loops and multiple invocations of model training and prediction, or we can use scikit-learn's cross_val_predict 

# pred_probs should be a Nx5 matrix of out-of-sample predicted probabilities, with N = len(data)
pred_probs = cross_val_predict(model, data, labels, method='predict_proba')

# Checking model accuracy on original data

preds = np.argmax(pred_probs, axis=1)
acc_original = accuracy_score(preds, labels)
print(f"Accuracy with original data: {round(acc_original*100,1)}%")

# Finding label issues automatically

# We count label issues using confident learning. First, we need to compute class thresholds for the different classes.
# Implement the Confident Learning algorithm for computing class thresholds for the 5 classes. 
# The class threshold for each class is the model's expected (average) self-confidence for each class. 
# In other words, to compute the threshold for a particular class, we can average the predicted probability for that class, for all datapoints that are labeled with that particular class.

def compute_class_thresholds(pred_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # this code is written in this style to make it easier to understand the algorithm
    # a more efficient implementation would use numpy vectorized operations and
    # scan over the data only once
    n_examples, n_classes = pred_probs.shape
    thresholds = np.zeros(n_classes)
    for k in range(n_classes):
        count = 0
        p_sum = 0
        for i in range(n_examples):
            if labels[i] == k:
                count += 1
                p_sum += pred_probs[i, k]
        thresholds[k] = p_sum / count
    return thresholds

# should be a numpy array of length 5
thresholds = compute_class_thresholds(pred_probs, labels.to_numpy())

# constructing the confident joint

def compute_confident_joint(pred_probs: np.ndarray, labels: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    # written using for loops to be understandable
    # this can be more efficiently implemented using numpy vectorized operations
    n_examples, n_classes = pred_probs.shape
    C = np.zeros((n_classes, n_classes), dtype=np.int)
    for data_idx in range(n_examples):
        i = labels[data_idx]
        j = None
        p_j = -1
        for candidate_j in range(n_classes):
            p = pred_probs[data_idx, candidate_j]
            if p >= thresholds[candidate_j] and p > p_j:
                j = candidate_j
                p_j = p
        if j is not None:
            C[i][j] += 1
    return C

C = compute_confident_joint(pred_probs, labels.to_numpy(), thresholds)

# count the number of label issues

num_label_issues = C.sum() - C.trace()

print('Estimated noise rate: {:.1f}%'.format(100*num_label_issues / pred_probs.shape[0]))

# filter out label issues

"""
In this lab, our approach to identifying issues is to rank the data points by a score ("self-confidence", the model's predicted probability for a data point's given label) and then take the top num_label_issues of those.

First, we want to compute the model's self-confidence for each data point. For a data point i, that is pred_probs[i, labels[i]].
"""

# this should be a numpy array of length 941 of probabilities
self_confidences = np.array([pred_probs[i, l] for i, l in enumerate(labels)])

# Next, we rank the indices of the data points by the self-confidence.

# this should be a numpy array of length 941 of integer indices
ranked_indices = np.argsort(self_confidences)

# Finally, let's compute the indices of label issues as the top num_label_issues items in the ranked_indices.

issue_idx = ranked_indices[:num_label_issues]

# Let's look at a couple of the highest-ranked data points (most likely to be label issues):
    
df_c.iloc[ranked_indices[:5]]

# How'd We Do?
# Let's go a step further and see how we did at automatically identifying which data points are mislabeled. 
# If we take the intersection of the labels errors identified by Confident Learning and the true label errors, we see that our approach was able to identify 75% of the label errors correctly (based on predictions from a model that is only 67% accurate).

# Computing percentage of true errors identified. 
true_error_idx = df[df.letter_grade != df.noisy_letter_grade].index.values
cl_acc = len(set(true_error_idx).intersection(set(issue_idx)))/len(true_error_idx)
print(f"Percentage of errors found: {round(cl_acc*100,1)}%")

# Train a More Robust Model

"""
Now that we have the indices of potential label errors within our data, let's remove them from our data, retrain our model, and see what improvement we can gain.

Keep in mind that our baseline model from above, trained on the original data using the noisy_letter_grade as the prediction label, achieved a cross-validation accuracy of 67%.

Let's use a very simple method to handle these label errors and just drop them entirely from the data and retrain our exact same XGBClassifier. In a real-world application, a better approach might be to have humans review the issues and correct the labels rather than dropping the data points.
"""

# Remove the label errors found by Confident Learning
data = df.drop(issue_idx)
clean_labels = data['noisy_letter_grade']
data = data.drop(['stud_ID', 'letter_grade', 'noisy_letter_grade'], axis=1)

# Train a more robust classifier with less erroneous data
model = XGBClassifier(tree_method="hist", enable_categorical=True)
clean_pred_probs = cross_val_predict(model, data, clean_labels, method='predict_proba')
clean_preds = np.argmax(clean_pred_probs, axis=1)

acc_clean = accuracy_score(clean_preds, clean_labels)
print(f"Accuracy with original data: {round(acc_original*100, 1)}%")
print(f"Accuracy with errors found by Confident Learning removed: {round(acc_clean*100, 1)}%")

# Compute reduction in error.
err = ((1-acc_original)-(1-acc_clean))/(1-acc_original)
print(f"Reduction in error: {round(err*100,1)}%")

"""
After removing the suspected label issues, our model's new cross-validation accuracy is now 90%, 
which means we reduced the error-rate of the model by 70% (the original model had 67% accuracy).

Note: throughout this entire process we never changed any code related to model architecture/hyperparameters, training, 
or data preprocessing! 
This improvement is strictly coming from increasing the quality of our data which leaves additional room for additional optimizations on the modeling side

"""

# Conclusion

# For the student grades dataset, we found that simply dropping identified label errors and retraining the model resulted in a 70% 
# reduction in prediction error on our classification problem (with accuracy improving from 67% to 90%).
# An implementation of the Confident Learning algorithm (and much more) is available in the cleanlab library on GitHub. 
# This is how today's lab assignment can be done in a single line of code with Cleanlab:
    
import cleanlab

cl_issue_idx = cleanlab.filter.find_label_issues(labels, pred_probs, return_indices_ranked_by='self_confidence')

df_c.iloc[cl_issue_idx[:5]]

"""
Advanced topic: you might notice that the above cl_issue_idx differs in length (by a little bit) from our issue_idx. The reason for this is that we implemented a slightly simplified version of the algorithm in this lab. We skipped a calibration step after computing the confident joint that makes the confident joint have the true noisy prior 
 (summed over columns for each row) and also add up to the total number of examples.
"""

