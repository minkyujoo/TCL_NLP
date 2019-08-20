import tensorflow as tf
import sys

from configs import DEFINES
import numpy as np

def layer_norm(inputs, eps=1e-6):
  feature_shape = inputs.get_shape()[-1:]
  mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
  std = tf.keras.backend.std(inputs,[-1], keepdims=True)
  beta = tf.get_variable("beta", initialize=tf.zeros(feature_shape))
  gamma = tf.get_variable("gamma", initialize=tf.zeros(feature_shape))
  return gamma * (inputs-mean) / (std+eps) + beta

def sublayer_connection(inputs, sublayer, dropout=2):
  outputs= layer_norm(inputs + tf.keras.Dropout(dropout)(sublayer))
  return outputs

