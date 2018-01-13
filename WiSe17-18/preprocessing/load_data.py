# -*- coding: utf-8 -*-

import tensorflow as tf
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
print(dir_path)
filename = dir_path + "/pos/en-train.txt"

tokens = tf.placeholder(tf.int32, name='tokens')
pos_tags = tf.placeholder(tf.string, name='pos_tags')
total = tf.reduce_sum(tokens, name='total')

printerop = tf.Print(total, [tokens, pos_tags], name='printer')

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    with open(filename) as inf:
        for line in inf:
            # Read data, using python, into our features
            token, pos_tag = line.strip().split("\t")
            token = str(token)
            pos_tag = str(pos_tag)
            # Run the Print ob
            total = sess.run(printerop, feed_dict={pos_tags: pos_tag, tokens: token})
            print(token, total)