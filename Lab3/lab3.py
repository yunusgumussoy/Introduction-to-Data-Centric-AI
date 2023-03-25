# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 18:58:26 2023

@author: Yunus

MIT Open Learning

Introduction to Data-Centric AI

Dataset Creation and Curation

Lab 3

Source:
    https://github.com/dcai-course/dcai-lab/blob/master/dataset_curation/solutions/Solutions%20-%20Dataset%20Curation.ipynb
"""

# pip install cleanlab
# We originally used the version: cleanlab==2.2.0
# This automatically installs other required packages like numpy, pandas, sklearn

import numpy as np
import pandas as pd

# Analyzing dataset labeled by multiple annotators

"""
We simulate a small classification dataset (3 classes, 2-dimensional features) with ground truth labels that are then hidden from our analysis. 
The analysis is conducted on labels from noisy annotators whose labels are derived from the ground truth labels, but with some probability of error in each annotated label where the probability is determined by the underlying quality of the annotator. 
In subsequent exercises, you should assume the ground truth labels and the true annotator qualities are unknown to you.

"""

## You don't need to understand this cell, it's just used for generating the dataset

SEED = 123  # for reproducibility
np.random.seed(seed=SEED)

def make_data(sample_size = 300):
    """ Produce a 3-class classification dataset with 2-dimensional features and multiple noisy annotations per example. """
    num_annotators=50  # total number of data annotators
    class_frequencies = [0.5, 0.25, 0.25]
    sizes=[int(np.ceil(freq*sample_size)) for freq in class_frequencies]  # number of examples belonging to each class
    good_annotator_quality = 0.6
    bad_annotator_quality = 0.3
    
    # Underlying statistics of the datset (unknown to you)
    means=[[3, 2], [7, 7], [0, 8]]
    covs=[[[5, -1.5], [-1.5, 1]], [[1, 0.5], [0.5, 4]], [[5, 1], [1, 5]]]
    
    m = len(means)  # number of classes
    n = sum(sizes)
    local_data = []
    labels = []

    # Generate features and true labels
    for idx in range(m):
        local_data.append(
            np.random.multivariate_normal(mean=means[idx], cov=covs[idx], size=sizes[idx])
        )
        labels.append(np.array([idx for i in range(sizes[idx])]))
    X_train = np.vstack(local_data)
    true_labels_train = np.hstack(labels)

    # Generate noisy labels from each annotator
    s = pd.DataFrame(
        np.vstack(
            [
                generate_noisy_labels(true_labels_train, good_annotator_quality)
                if i < num_annotators - 10  # last 10 annotators are worse
                else generate_noisy_labels(true_labels_train, bad_annotator_quality)
                for i in range(num_annotators)
            ]
        ).transpose()
    )

    # Each annotator only labels approximately 10% of the dataset (unlabeled points represented with NaN)
    s = s.apply(lambda x: x.mask(np.random.random(n) < 0.9)).astype("Int64")
    s.dropna(axis=1, how="all", inplace=True)
    s.columns = ["A" + str(i).zfill(4) for i in range(1, num_annotators+1)]
    # Drop rows not annotated by anybody
    row_NA_check = pd.notna(s).any(axis=1)
    X_train = X_train[row_NA_check]
    true_labels_train = true_labels_train[row_NA_check]
    multiannotator_labels = s[row_NA_check].reset_index(drop=True)
    # Shuffle the rows of the dataset
    shuffled_indices = np.random.permutation(len(X_train))
    return {
        "X_train": X_train[shuffled_indices],
        "true_labels_train": true_labels_train[shuffled_indices],
        "multiannotator_labels": multiannotator_labels.iloc[shuffled_indices],
    }

def generate_noisy_labels(true_labels, annotator_quality):
    """ Randomly flips each true label to a different class with probability that depends on annotator_quality. """
    n = len(true_labels)
    m = np.max(true_labels) + 1  # number of classes
    annotated_labels = np.random.randint(low=0, high=3, size=n)
    correctly_labeled_indices = np.random.random(n) < annotator_quality
    annotated_labels[correctly_labeled_indices] = true_labels[correctly_labeled_indices]
    return annotated_labels

data_dict = make_data(sample_size = 300)

X = data_dict["X_train"]
multiannotator_labels = data_dict["multiannotator_labels"]
true_labels = data_dict["true_labels_train"] # used for comparing the accuracy of consensus labels


"""
Let's view the first few rows of the data used for this exercise. 
Here are the labels selected by each annotator for the first few examples. 
Here each example is a row, and the annotators are columns. Not all annotators labeled each example; 
valid class annotations from those that did label the example are integers (either 0, 1, or 2 for our 3 classes), 
and otherwise the annotation is left as NA if a particular annotator did not label a particular example.
"""

multiannotator_labels.head()

# Here are the corresponding 2D data features for these examples:
X[:5]

# Train model with cross-validation

# In this exercise, we consider the simple K Nearest Neighbors classification model, 
# which produces predicted class probabilities for a particular example via a (weighted) 
# average of the labels of the K closest examples. 
# We will train this model via cross-validation and use it to produce held-out predictions for each example in our dataset.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict

def train_model(labels_to_fit):
    """ Trains a simple feedforward neural network model on the data features X with y = labels_to_fit, via cross-validation.
        Returns out-of-sample predicted class probabilities for each example in the dataset
        (from a copy of model that was never trained on this example).
        Also evaluates the held-out class predictions against ground truth labels.
    """
    num_crossval_folds = 5  # number of folds of cross-validation
    # model = MLPClassifier(max_iter=1000, random_state=SEED)
    model = KNeighborsClassifier(weights="distance")
    pred_probs = cross_val_predict(
        estimator=model, X=X, y=labels_to_fit, cv=num_crossval_folds, method="predict_proba"
    )
    class_predictions = np.argmax(pred_probs, axis=1)
    held_out_accuracy = np.mean(class_predictions == true_labels)
    print(f"Accuracy of held-out model predictions against ground truth labels: {held_out_accuracy}")
    return pred_probs

"""
Here we demonstrate how to train and evaluate this model. 
Note that the evaluation is against ground truth labels, which you wouldn't have in real applications, 
so this evaluation is just for demonstration purposes. 
We'll first fit this model using labels comprised of one randomly selected annotation for each example.
"""

labels_from_random_annotators = true_labels.copy()
for i in range(len(multiannotator_labels)):
    annotations_for_example_i = multiannotator_labels.iloc[i][pd.notna(multiannotator_labels.iloc[i])]
    labels_from_random_annotators[i] = np.random.choice(annotations_for_example_i.values)

print(f"Accuracy of random annotators' labels against ground truth labels: {np.mean(labels_from_random_annotators == true_labels)}")
pred_probs_from_model_fit_to_random_annotators = train_model(labels_to_fit = labels_from_random_annotators)

# We can also fit this model using the ground truth labels (which you would not be able to in practice), just to see how good it could be:

pred_probs_from_unrealistic_model_fit_to_true_labels = train_model(labels_to_fit = true_labels)

# Exercise 1
"""
Compute majority-vote consensus labels for each example from the data in multiannotator_labels. Think about how to best break ties!

Evaluate the accuracy of these majority-vote consensus labels against the ground truth labels.
Also set these as labels_to_fit in train_model() to see the resulting model's accuracy when trained with majority vote consensus labels.
Estimate the quality of annotator (how accurate their labels tend to be overall) using only these majority-vote consensus labels (assume the ground truth labels are unavailable as they would be in practice). Who do you guess are the worst 10 annotators?
"""
## Solution to Exercise 1. 
## Uses cleanlab library: https://docs.cleanlab.ai/stable/tutorials/multiannotator.html
## See the source code for implementation: https://github.com/cleanlab/cleanlab/blob/master/cleanlab/multiannotator.py

from cleanlab.multiannotator import get_majority_vote_label

majority_vote_labels = get_majority_vote_label(multiannotator_labels)
print(f"Accuracy of majority-vote consensus labels against ground truth labels: {np.mean(majority_vote_labels == true_labels)}")
pred_probs_from_model_fit_to_majority_vote_labels = train_model(labels_to_fit = majority_vote_labels)

annotator_quality_estimates = np.zeros(multiannotator_labels.shape[1],)
for annotator_index in range(multiannotator_labels.shape[1]):
    annotator_labels = multiannotator_labels.iloc[:, annotator_index]
    labeled_examples = pd.notna(annotator_labels)
    annotator_quality_estimates[annotator_index] = np.mean(annotator_labels.values[labeled_examples] == majority_vote_labels[labeled_examples])

print(f"Worst 10 annotators are inferred to be: {[multiannotator_labels.columns[i] for i in np.argsort(annotator_quality_estimates)[:10]]}")

# The true answer is the last 10 annotators A0041-A0050 are the lowest quality annotators but one cannot guarantee this can be accurately estimated from data.

# Exercise 2
"""
Estimate consensus labels for each example from the data in multiannotator_labels, this time using the CROWDLAB algorithm. You may find it helpful to reference: https://docs.cleanlab.ai/stable/tutorials/multiannotator.html Recall that CROWDLAB requires out of sample predicted class probabilities from a trained classifier. You may use the pred_probs from your model trained on majority-vote consensus labels or our randomly-selected annotator labels. Which do you think would be better to use?

Evaluate the accuracy of these CROWDLAB consensus labels against the ground truth labels.
Also set these as labels_to_fit in train_model() to see the resulting model's accuracy when trained with CROWDLAB consensus labels.
Estimate the quality of annotator (how accurate their labels tend to be overall) using CROWDLAB (assume the ground truth labels are unavailable as they would be in practice). Who do you guess are the worst 10 annotators based on this method?
"""

## Solution to Exercise 2.
## Uses cleanlab library: https://docs.cleanlab.ai/stable/tutorials/multiannotator.html
## See the source code for implementation: https://github.com/cleanlab/cleanlab/blob/master/cleanlab/multiannotator.py


from cleanlab.multiannotator import get_label_quality_multiannotator

# We use the predicted class probabilities from classifier trained on majority vote labels, 
# since those are more accurate than the predicitions from classifier trained on random annotators' labels.
pred_probs = pred_probs_from_model_fit_to_majority_vote_labels  # alternatively: pred_probs_from_model_fit_to_random_annotators
results = get_label_quality_multiannotator(multiannotator_labels, pred_probs, verbose=False)
crowdlab_labels = results["label_quality"]["consensus_label"]

print(f"Accuracy of CROWDLAB consensus labels against ground truth labels: {np.mean(crowdlab_labels == true_labels)}")
pred_probs_from_model_fit_to_random_annotators = train_model(labels_to_fit = crowdlab_labels)

annotator_quality_estimates = results["annotator_stats"]
print(f"Worst 10 annotators are inferred to be: {annotator_quality_estimates.index[:10].tolist()}")

# The true answer is the last 10 annotators A0041-A0050 are the lowest quality annotators but one cannot guarantee this can be accurately estimated from data.
