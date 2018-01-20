# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf

class Preprocessing:
    def __init__(self):
        filename = '../glove.6B.50d.txt'
        # Load Glove pretrained vocabulary
        embedding_vocabulary = self.loadGloVe(filename)
        vocab_size = len(embedding_vocabulary)
        embedding_dim = len(list((embedding_vocabulary.values()))[0])
        embedding = np.asarray(embedding_vocabulary.values())

        W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)
        
        print(embedding_vocabulary['the'])

        init_op = tf.global_variables_initializer()
        # init_op = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init_op)
            print(sess.run(W))
        
    def loadGloVe(self, filename):
        embedding_dict = dict()
        file = open(filename,'r')
        for line in file.readlines():
            row = line.split(" ")
            embedding_dict[row[0]] = [float(i) for i in row[1:]]
        
        print('Loaded GloVe!')
        file.close()
        return embedding_dict

if __name__ == "__main__":
    Preprocessing()


# ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(ROOT_DIR, "../pos/en-train.txt")

# tokens = tf.placeholder(tf.int32, name='tokens')
# pos_tags = tf.placeholder(tf.string, name='pos_tags')
# total = tf.reduce_sum(tokens, name='total')

# printerop = tf.Print(total, [tokens, pos_tags], name='printer')

# with tf.Session() as sess:
#     sess.run( tf.global_variables_initializer())
#     with open(DATA_DIR) as inf:
#         for line in inf:
#             if line.strip():
#                 # Read data, using python, into our features
#                 token, pos_tag = line.strip().split("\t")
#                 token = str(token)
#                 pos_tag = str(pos_tag)
#                 # Run the Print ob
#                 print(token, pos_tag)