# -*- coding: utf-8 -*-
# Team LSD-M
# Purpose: RNN implementation for NER tagger via TensorFlow

import tensorflow as tf
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR + '/../')
from preprocessing.load_data_glove import Preprocessing
p = Preprocessing()