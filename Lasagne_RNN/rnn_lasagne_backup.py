#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Recurrent network example.  Trains a bidirectional vanilla RNN to output the
sum of two numbers in a sequence of random numbers sampled uniformly from
[0, 1] based on a separate marker sequence.
'''

from __future__ import print_function
from create_dataset import *

import numpy as np
import theano
import theano.tensor as T
import lasagne


# Min/max sequence length
MIN_LENGTH = 50
MAX_LENGTH = 55
# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 10


def gen_data(n_batch=N_BATCH):
  print(n_batch)
  # Generate X - we'll fill the last dimension later
  X, y, num_units = preprocess("../pos/2/de-train.txt")
  #for n in range(n_batch):
  # Center the inputs and outputs
  return (X, y, num_units)

def main(num_epochs=NUM_EPOCHS):
  print("Building network ...")

  # Load data
  X_val, y_val, num_units = gen_data()

  # First, we build the network, starting with an input layer
  # Recurrent layers expect input of shape
  # (batch size, max sequence length, number of features)
  l_in = lasagne.layers.InputLayer((None, X_val.shape[1]), input_var=X_val)
  # We're using a bidirectional network, which means we will combine two
  # RecurrentLayers, one with the backwards=True keyword argument.
  # Setting a value for grad_clipping will clip the gradients in the layer
  # Setting only_return_final=True makes the layers only return their output
  # for the final time step, which is all we need for this task
  l_forward = lasagne.layers.RecurrentLayer(
      l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
      W_in_to_hid=lasagne.init.HeUniform(),
      W_hid_to_hid=lasagne.init.HeUniform(),
      nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
  l_backward = lasagne.layers.RecurrentLayer(
      l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
      W_in_to_hid=lasagne.init.HeUniform(),
      W_hid_to_hid=lasagne.init.HeUniform(),
      nonlinearity=lasagne.nonlinearities.tanh,
      only_return_final=True, backwards=True)
  # Now, we'll concatenate the outputs to combine them.
  l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
  # Our output layer is a simple dense connection, with 1 output unit
  l_out = lasagne.layers.DenseLayer(
      l_concat, num_units=num_units, nonlinearity=lasagne.nonlinearities.softmax)

  X_train = T.fmatrix('input_train')
  target_values = T.ivector('output_train')

  # lasagne.layers.get_output produces a variable for the output of the net
  network_output = lasagne.layers.get_output(l_out)
  # The network output will have shape (n_batch, 1); let's flatten to get a
  # 1-dimensional vector of predicted values
  predicted_values = network_output.flatten()
  print(predicted_values)
  # Our cost will be mean-squared error
  cost = T.mean((predicted_values - target_values)**2)
  # Retrieve all parameters from the network
  all_params = lasagne.layers.get_all_params(l_out)
  # Compute SGD updates for training
  print("Computing updates ...")
  updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
  # Theano functions for training and computing cost
  print("Compiling functions ...")
  train = theano.function(inputs=[X_train, target_values],
                          outputs=[predicted_values, cost],
                          updates=updates,
                          allow_input_downcast=True)

  compute_cost = theano.function(
      [X_train, target_values], cost)

  print("Training ...")
  try:
      for epoch in range(num_epochs):
          for _ in range(EPOCH_SIZE):
              X, y, num_units = gen_data()
              train(X, y)
          cost_val = compute_cost(X_val, y_val)
          print("Epoch {} validation cost = {}".format(epoch, cost_val))
  except KeyboardInterrupt:
      pass

if __name__ == '__main__':
    main()