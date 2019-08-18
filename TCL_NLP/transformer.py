
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt

#transformer model for tensorflow 2.0
#from __future__ import absolute_import, division, print_function, unicode_literals
#!pip install -q tensorflow-gpu==2.0.0-beta1

def get_angles(pos, i, d_model):
    angle_rates = 1/np.power(10000, (2(i/2))/np.float32(d_model))
    return pos *angle_rates

def positional_encoding (position, d_model ):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
    np.arrange(d_model)[np.newaxis, :], d_model)

    sines = tf.math.sin(angle_rads[:,0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def scaled_dot_product_attention(query, key, value, mask):
    #attention score matrix = matmul (Q , K)
    matmul_qk= tf.matmul(query, key, transpose_b = True)

    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk /tf.math.sqrt(dk)

    if mask is not None:
        logits += (mask*-1e9)

    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights

#test code for scaled_out_product_attention
np.set_printoptions(suppress = True)
tmp_k = tf.constant([[10,0,0], 
                     [0,10,0],
                     [0,0,10],
                     [0,0,10]], dtype=tf.float32)
tmp_v = tf.constant([[1,0],
                     [10,0],
                     [100,5],
                     [1000,6]], dtype=tf.float32)
tmp_q = tf.conconstant([[0,10,0]],dtype=tf.float32)
temp_out, temp_attn = scaled_dot_product_attention(tmp_q, tmp_k, tmp_v, None)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads ==0
        self.depth = d_model 
        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)
        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0,2,1,3])

    def call (self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0,2,1,3])

        concat_attention = tf.reshape(scaled_attentionext, (batch_size, -1, self.d_model))
        outputs= self.dense(concat_attention)
        return outputs




