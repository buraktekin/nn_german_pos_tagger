#! /usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
import locale
#from utils import *
from rnn import RNN

locale.setlocale(locale.LC_ALL, 'de_DE')

# Read data from file
words_list = list()
tags_list = list()
with open('./pos/2/de-train.txt', 'rb') as f_words:
	print "Loading the training data..."
	for word in f_words:
		if word == "\n":
			pass
		else:
			words_list.append(word.split()[0].replace("\n", "").decode("utf-8"))
			tags_list.append(word.split()[1])

	print "Training data has just loaded."
	print "=============================="
	f_words.close()

# Find unique words, tokens.
print "Word Frequency Measuring..."
word_freq = nltk.FreqDist(words_list)
vocabulary_size = len(word_freq.items())
print "Found %d unique words tokens." % vocabulary_size
print "=============================="

# Find unique tags.
print "Word Frequency Measuring..."
tag_freq = nltk.FreqDist(tags_list)
tag_size = len(tag_freq.items())
print "Found %d unique tags." % tag_size
print "=============================="

# Options.
_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', vocabulary_size))
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '80'))
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
_NEPOCH = int(os.environ.get('NEPOCH', '100'))
_MODEL_FILE = os.environ.get('MODEL_FILE')

# Generate vector of words.
word_tokens = word_freq.most_common(vocabulary_size)
unique_words = [x[0] for x in word_tokens]
tag_tokens = tag_freq.most_common(tag_size)
unique_tags = [t[0] for t in tag_tokens]
#dictionaries of (word/tags, integer_assigned_word) -> dict of tuples
word_and_index = dict([(w,i+len(unique_tags)) for i,w in enumerate(unique_words)])
tag_and_index = dict([(w,i) for i,w in enumerate(unique_tags)])
print "Using vocabulary size %d." % vocabulary_size
print "=============================="

# Create the training data
X_train = np.asarray([word_and_index[word] for word in words_list])
y_train = np.asarray([tag_and_index[tag] for tag in tags_list])

print X_train
print y_train
print "X: ", len(X_train)
print "Y: ", len(y_train)

np.random.seed(10)
model = RNN(tag_size)
o, s = model.predict_word_probabilities(y_train)
print o.shape
print o

predictions = model.predict([X_train[10]])
print predictions.shape
print predictions

