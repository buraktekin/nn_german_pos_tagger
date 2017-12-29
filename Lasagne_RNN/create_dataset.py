__author__ = 'csburak'
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import nltk
import itertools
from gensim.models import Word2Vec

def preprocess(filename):
  # Read data from file
  words_list = list() # tokenized sentences
  tags_list = list() # tokenized tags
  full_word_list = list()
  full_tag_list = list()
  with open(filename) as f_words:
    print "Loading..."
    temp_words = list()
    temp_tags = list()
    for word in f_words:
      if word == "\n":
        words_list.append(temp_words)
        tags_list.append(temp_tags)
        temp_words, temp_tags = [], []
        pass
      else:
        w_t = word.split()[0].replace("\n", "").decode("utf-8")
        t_t = word.split()[1]
        temp_words.append(w_t)
        temp_tags.append(t_t)
        full_word_list.append(w_t)
        full_tag_list.append(t_t)
  print "Training data has just loaded."
  print "========================================"
  f_words.close()

  # Create Word2Vec model
  model_w2v = Word2Vec(words_list, min_count=1, size=100)

  # Vector representations of each words
  word_vectors = np.array([model_w2v[w].flatten() for w in full_word_list])
  # Find unique tags.
  print "Tag Frequency Measuring..."
  tag_freq = nltk.FreqDist(itertools.chain(*tags_list))
  tag_size = len(tag_freq.items())
  print "Found %d unique tags." % tag_size
  print "========================================"

  tag_tokens = tag_freq.most_common(tag_size)
  unique_tags = [t[0] for t in tag_tokens]
  # #dictionaries of (word/tags, integer_assigned_word) -> dict of tuples
  tag_and_index = dict([(w,i) for i,w in enumerate(unique_tags)])

  # Create the training data
  tag_vectors = np.array([tag_and_index[t] for t in full_tag_list])

  return (word_vectors, tag_vectors, tag_size)