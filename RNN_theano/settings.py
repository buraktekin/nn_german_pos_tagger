#! /usr/bin/env python
# -*- coding: utf-8 -*-

import locale, os

# Set Language options for better encoding
locale.setlocale(locale.LC_ALL, 'de_DE')

HIDDEN_LAYER_SIZE = int(os.environ.get('HIDDEN_LAYER_SIZE', '200'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.05'))
NUMBER_OF_EPOCH = int(os.environ.get('NUMBER_OF_EPOCH', '100'))
CURRENT_MODEL = os.environ.get('CURRENT_MODEL')