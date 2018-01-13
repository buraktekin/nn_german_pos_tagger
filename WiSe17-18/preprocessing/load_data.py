# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf


filename = '../glove.6B.50d.txt'

def loadGloVe(filename):
    embedding_dict = dict()
    file = open(filename,'r')
    for line in file.readlines():
        row = line.split(" ")
        embedding_dict[row[0]] = [float(i) for i in row[1:]]
    
    print('Loaded GloVe!')
    file.close()
    return embedding_dict

# Load Glove pretrained vocabulary
embd = loadGloVe(filename)
vocab_size = len(embd)
embedding_dim = len(list((embd.values()))[0])
embedding = np.asarray(embd.values)

W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
embedding_init = W.assign(embedding_placeholder)

print(embd['the'])

init_op = tf.global_variables_initializer()
# init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(W))
    # init.run()
    # sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    # tf.Print(W[0], vocab)
    # for i in vocab[:10]:
    #     print(i)
    # print(embedding[0])

# from tensorflow.contrib import learn
# #init vocab processor
# vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
# #fit the vocab from glove
# pretrain = vocab_processor.fit(vocab)
# #transform inputs
# x = np.array(list(vocab_processor.transform(your_raw_input)))










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