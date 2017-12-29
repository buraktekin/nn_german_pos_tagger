__author__ = 'csburak'

from create_dataset import *

import numpy
import theano
import theano.tensor as T
import lasagne

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 80
# Number of training sequences in each batch
N_BATCH = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# How often should we check the output?
EPOCH_SIZE = 100
# Number of epochs to train the net
NUM_EPOCHS = 100

X_train, y_train, tag_size = preprocess("../pos/2/de-train.txt")
num_unit = tag_size
N = X_train.shape[0] # training sample size: 719530
feats = X_train.shape[1] # number of input: 100

def build_network(input_var):
  network = lasagne.layers.InputLayer((N, feats), input_var=input_var)
  l_forward = lasagne.layers.RecurrentLayer(
    network, N_HIDDEN,
    W_in_to_hid=lasagne.init.Uniform(),
    W_hid_to_hid=lasagne.init.Uniform(),
    nonlinearity=lasagne.nonlinearities.tanh,
    only_return_final=True)

  l_backward = lasagne.layers.RecurrentLayer(
    network, N_HIDDEN,
    W_in_to_hid=lasagne.init.Uniform(),
    W_hid_to_hid=lasagne.init.Uniform(),
    nonlinearity=lasagne.nonlinearities.tanh,
    only_return_final=True, backwards=True)

  # Now, we'll concatenate the outputs to combine them.
  l_concat = lasagne.layers.ConcatLayer([l_forward, l_backward])
  # Our output layer is a simple dense connection, with 1 output unit
  network = lasagne.layers.DenseLayer(l_concat, num_units=num_unit, nonlinearity=lasagne.nonlinearities.softmax)
  return network

def main():
  # generate a dataset: D = (input_values, target_class)
  D = (X_train, y_train)
  training_steps = 1000

  # Declare Theano symbolic variables
  x = T.fmatrix('x')
  y = T.ivector('y')

  print "seks"
  network = build_network(x)
  print "seks"
  output = lasagne.layers.get_output(network)
  prediction = output.flatten()
  loss = T.mean((prediction - y)**2)

  params = lasagne.layers.get_all_params(network)
  # add regularization
  l2_penalty = lasagne.regularization.regularize_layer_params(lasagne.layers.get_all_layers(network), lasagne.regularization.l2) * 0.01
  loss = loss + l2_penalty
  updates = lasagne.updates.sgd(loss, params, learning_rate=0.01)
  # Compile
  train = theano.function(
          inputs=[x, y],
          outputs=[prediction, loss],
          updates=updates,
          allow_input_downcast=True)

  # Train
  for i in range(training_steps):
      pred, err = train(D[0], D[1])
      print "Epoch #{}:".format(str(i))
      print "training error " + str(err)

if __name__ == '__main__':
    main()