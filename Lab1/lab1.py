# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 02:12:48 2023

@author: Yunus

MIT Open Learning

Introduction to Data-Centric AI

Data-Centric AI vs. Model-Centric AI

Lab 1

Source:
    https://github.com/dcai-course/dcai-lab/blob/master/data_centric_model_centric/solutions/Solution%20-%20Data-Centric%20AI%20vs%20Model-Centric%20AI.ipynb
"""

# pip install scikit-learn pandas

import pandas as pd

train = pd.read_csv('reviews_train.csv')
test = pd.read_csv('reviews_test.csv')

print(test.sample(5))

# Training a baseline model

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

sgd_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier()),
])

_ = sgd_clf.fit(train['review'], train['label'])

# Evaluating model accuracy

from sklearn import metrics

def evaluate(clf):
    pred = clf.predict(test['review'])
    acc = metrics.accuracy_score(test['label'], pred)
    print(f'Accuracy: {100*acc:.1f}%')


evaluate(sgd_clf)

# 76% accuracy is not great for this binary classification problem.

# Trying another model

# Naive Bayes model

from sklearn.naive_bayes import MultinomialNB

nb_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

nb_clf.fit(train['review'], train['label'])

evaluate(nb_clf)

# Accuracy: 85.3%. Better!

# Taking a closer look at the training data

print(train.head())

# Zooming in on one particular data point:

print(train.iloc[0].to_dict())

# It looks like there's some funny HTML tags in our dataset, and those datapoints have nonsense labels.

"""
To address this, a simple approach we might try is to throw out the bad data points, and train our model on only the "clean" data.

Come up with a simple heuristic to identify data points containing HTML, and filter out the bad data points to create a cleaned training set.
"""

def is_bad_data(review: str) -> bool:
    # a simple heuristic, but it works pretty well;
    # finds all HTML tags, though there might be some
    # false positives
    return '<' in review

# Creating the cleaned training set

train_clean = train[~train['review'].map(is_bad_data)]

# Evaluating a model trained on the clean training set

from sklearn import clone

sgd_clf_clean = clone(sgd_clf)

_ = sgd_clf_clean.fit(train_clean['review'], train_clean['label'])

evaluate(sgd_clf_clean)

# Accuracy: 97.0%. Much Better!

# Training a Transformer model with HuggingFace

# pip install torch transformers datasets
# pip install --upgrade protobuf

# After installing all these, if you still experience issues with the following dependencies, consider using a Colab notebook instead of your own laptop.

import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments

import datasets
from datasets import Dataset, DatasetDict, ClassLabel

# Reformat the data to be suitable with the HuggingFace Dataset class

label_map = {"bad": 0, "good": 1}
dataset_train = Dataset.from_dict({"label": train["label"].map(label_map), "text": train["review"].values})
dataset_test = Dataset.from_dict({"label": test["label"].map(label_map), "text": test["review"].values})

model_name = "distilbert-base-uncased"  # which pretrained neural network weights to load for fine-tuning on our data
# other options you could try: "bert-base-uncased", "bert-base-cased", "google/electra-small-discriminator"

max_training_steps = 10  # how many iterations our network will be trained for
# Here set to a tiny value to ensure quick runtimes, set to higher values if you have a GPU to run this code on.

model_folder = "test_trainer"  # file where model will be saved after training

# Now we can train our Transformer model with the configuration selected above.

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_tokenized_dataset = dataset_train.map(tokenize_function, batched=True)
train_tokenized_dataset = train_tokenized_dataset.cast_column("label", ClassLabel(names = ["0", "1"]))

test_tokenized_dataset = dataset_test.map(tokenize_function, batched=True)
test_tokenized_dataset = test_tokenized_dataset.cast_column("label", ClassLabel(names = ["0", "1"]))

training_args = TrainingArguments(max_steps=max_training_steps, output_dir=model_folder)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized_dataset,
)

trainer.train()  # may take a while to train (try to run on a GPU if you can access one)

# Finally we evaluate the Transformer network's accuracy on our test data

pred_probs = trainer.predict(test_tokenized_dataset).predictions
pred_classes = np.argmax(pred_probs, axis=1)
print(f"Error rate of predictions: {np.mean(pred_classes != test_tokenized_dataset['label'])}")


