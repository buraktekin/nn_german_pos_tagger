from utils import *
from theano import tensor as T

import numpy as np
import theano as theano
import operator

class RNN:
  def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
    # Assign instance variables
    self.word_dim = word_dim
    self.hidden_dim = hidden_dim
    self.bptt_truncate = bptt_truncate
    # Randomly initialize the network parameters
    U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
    V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
    W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
    # Theano: Created shared variables
    self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
    self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
    self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
    # We store the Theano graph here
    self.theano = {}
    self.__theano_build__()

  def __theano_build__(self):
    U, V, W = self.U, self.V, self.W
    x = T.fmatrix('x')
    y = T.ivector('y')
    def forward_prop_step(x_t, s_t_prev, U, V, W):
      s_t = T.tanh(U[:,200] + W.dot(s_t_prev))
      #s_t = T.tanh(T.dot(U, x_t) + T.dot(W, s_t_prev))
      o_t = T.nnet.softmax(V.dot(s_t))
      return [o_t[0], s_t]

    [o,s], updates = theano.scan(
        forward_prop_step,
        sequences=x,
        outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
        non_sequences=[U, V, W],
        truncate_gradient=self.bptt_truncate,
        strict=True)

    prediction = T.argmax(o, axis=1)
    o_error = T.sum(T.nnet.categorical_crossentropy(o, y))

    # Gradients
    dU = T.grad(o_error, U)
    dV = T.grad(o_error, V)
    dW = T.grad(o_error, W)

    # Assign functions
    self.forward_propagation = theano.function([x], [o[-1], s[-1]])
    self.predict = theano.function([x], prediction)
    self.ce_error = theano.function([x, y], o_error, allow_input_downcast=True)
    self.bptt = theano.function([x, y], [dU, dV, dW])

    # SGD
    learning_rate = T.scalar('learning_rate')
    self.sgd_step = theano.function([x,y,learning_rate], [],
                  updates=[(self.U, self.U - learning_rate * dU),
                          (self.V, self.V - learning_rate * dV),
                          (self.W, self.W - learning_rate * dW)],
                                    allow_input_downcast=True)

  def calculate_total_loss(self, X, Y):
    return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])

  def calculate_loss(self, X, Y):
    # Divide calculate_loss by the number of words
    num_words = np.sum([1 for _ in Y])
    return self.calculate_total_loss(X,Y)/float(num_words)