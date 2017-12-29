#! /usr/bin/env python
# -*- coding: utf-8 -*-

from rnn import *
from settings import *
import sys
import numpy as np
import nltk
import itertools
from gensim.models import Word2Vec

def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

def save_model(outfile, model):
    U, V, W = model.U.get_value(), model.V.get_value(), model.W.get_value()
    np.savez(outfile, U=U, V=V, W=W)
    print "Saved model parameters to %s." % outfile
    print "================================================================\n"

def load_model(path, model):
  model_file = np.load(path)
  U, V, W = model_file["U"], model_file["V"], model_file["W"]
  model.hidden_dim = U.shape[0]
  model.word_dim = U.shape[1]
  model.U.set_value(U)
  model.V.set_value(V)
  model.W.set_value(W)
  print "Loaded model parameters from %s. hidden_dim=%d word_dim=%d" % (path, U.shape[0], U.shape[1])
  print "================================================================\n"

def predict_tags(model, input, output, learning_rate):
  temp_true_tagged = 0
  temp_false_tagged = 0
  for index in range(len(input)):
      true_tagged = 0
      false_tagged = 0
      o = model.forward_propagation(input[index])
      predictions = model.predict(input[index])
      y_train_np = np.asarray(output[index])

      for a in range(len(predictions)):
          if predictions[a] == y_train_np[a]:
              # Partial accuracy for each prediction
              temp_true_tagged += 1
          else:
              temp_false_tagged += 1
      true_tagged += temp_true_tagged
      false_tagged += temp_false_tagged

      print predictions.shape
      print "Pre: ", predictions
      print "Org: ", y_train_np
      # Total accuracy out of all tests
      print temp_true_tagged, temp_false_tagged

  print "True tagged: %d, false_tagged %d" % (true_tagged, false_tagged)
  print "accuracy: %.2f%s" % (100. * true_tagged / (true_tagged + false_tagged), "%")
  print "================================================================\n"

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
  model_w2v = Word2Vec(words_list, min_count=1, size=200)

  # Find unique words, tokens.
  print "Word Frequency Measuring..."
  word_freq = nltk.FreqDist(itertools.chain(*words_list))
  lexicon_size = len(word_freq.items())
  print "Found %d unique words tokens." % lexicon_size
  print "========================================"

  # Find unique tags.
  print "Tag Frequency Measuring..."
  tag_freq = nltk.FreqDist(itertools.chain(*tags_list))
  tag_size = len(tag_freq.items())
  print "Found %d unique tags." % tag_size
  print "========================================"

  # # Generate vector of words.
  word_tokens = word_freq.most_common(lexicon_size)
  unique_words = [w[0] for w in word_tokens]
  tag_tokens = tag_freq.most_common(tag_size)
  unique_tags = [t[0] for t in tag_tokens]
  # #dictionaries of (word/tags, integer_assigned_word) -> dict of tuples
  word_to_index = dict([(w,i) for i,w in enumerate(unique_words)])
  tag_to_index = dict([(t,i) for i,t in enumerate(unique_tags)])
  print "Using lexicon size %d." % lexicon_size
  print "========================================"

  # Create the training data
  X_s = np.array([[model_w2v[w].flatten() for w in word] for word in words_list])
  #y = np.array([[tag_to_index[t] for t in tag] for tag in tags_list])

  print "corpus", len(full_word_list)
  print "sentences:", len(X_s)

  X = np.asarray([[model_w2v[w].flatten()] for w in full_word_list])
  #X = np.array([[model_w2v[w].flatten() for w in word] for word in words_list])
  y = np.asarray([tag_to_index[t] for t in full_tag_list]).reshape((1,-1))
  return (X, y, lexicon_size)