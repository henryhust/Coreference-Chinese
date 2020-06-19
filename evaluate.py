#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import util

def read_doc_keys(fname):
    keys = set()
    with open(fname) as f:
        for line in f:
            keys.add(line.strip())
    return keys

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = util.get_model(config)
  saver = tf.train.Saver()
  log_dir = config["log_dir"]
  with tf.Session() as session:
    model.restore(session)
    # Make sure eval mode is True if you want official conll results
    test_summary, test_f, test_p, test_r, test_men_r = model.evaluate(session, mod="test")
    print("test_f1={}, test_p={},test_r={},test_men_r={}".format(test_f, test_p, test_r, test_men_r))
    