import numpy as np
import tensorflow as tf
from controller import Controller


"""
A 1-Layer recurrent neural network (LSTM) with 64 hidden nodes
"""

class RNNController(Controller):

    def init_controller_params(self):
        rnn_dim = 64
        init = tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32)

        self.params['cell'] = tf.nn.rnn_cell.BasicLSTMCell(rnn_dim, initializer = init)
        self.params['state'] = tf.Variable(tf.zeros([self.batch_size, rnn_dim]), trainable=False)
        self.params['output'] = tf.Variable(tf.zeros([self.batch_size, rnn_dim]), trainable=False)


    def nn_step(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.params['cell'](X, state)

    def zero_state(self):
        return (self.params['output'], self.params['state'])
