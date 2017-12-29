#! /usr/bin/env python
# -*- coding: utf-8 -*-

from utils import *
from settings import *
from datetime import datetime

import glob
import matplotlib.pyplot as plt
import sys
import time

train_inputs, train_outputs, lexicon_size = preprocess("../pos/2/de-train.txt")

def train_with_sgd(model, X_train, y_train, learning_rate=0.005, epoch=1, evaluate_loss_after=5):
    # We keep track of the losses so we can plot them later
    losses = []
    epochs = []
    num_examples_seen = 0
    for epoch in range(epoch):
      # Optionally evaluate the loss
      if (epoch % evaluate_loss_after == 0):
        epochs.append(epoch)
        loss = model.calculate_loss(X_train, y_train)
        losses.append((num_examples_seen, loss))
        time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        print "%s: Loss after num_examples_seen=%d epoch=%d: %f" % (time, num_examples_seen, epoch, loss)
        # Adjust the learning rate if loss increases
        if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
            learning_rate = learning_rate * 0.5
            print "Setting learning rate to %f" % learning_rate
        sys.stdout.flush()
        # ADDED! Saving model oarameters
        save_model("./rnn_models/rnn-theano-%d-%d-%s.npz" % (model.hidden_dim, model.word_dim, time), model)
      # For each training example...
      for i in range(len(y_train)):
          # One SGD step
          model.sgd_step(X_train[i], y_train[i], learning_rate)
          num_examples_seen += 1

model = RNN(lexicon_size, hidden_dim=HIDDEN_LAYER_SIZE)

print "==========================================="
models = glob.glob("./rnn_models/rnn-*")
if len(models) > 0:
  print models
  print "A model file found. Loading parameters..."
  CURRENT_MODEL = models[-1]
  load_model(CURRENT_MODEL, model)
print "==========================================="

train_with_sgd(model,
               train_inputs,
               train_outputs,
               epoch=NUMBER_OF_EPOCH,
               learning_rate=LEARNING_RATE)

test_input, test_output, tag_size = preprocess("../pos/2/de-dev.txt")
predict_tags(model, test_input, test_output, LEARNING_RATE)