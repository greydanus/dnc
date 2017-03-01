import numpy as np
import tensorflow as tf
from controller import Controller
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import LSTMStateTuple

"""
RNN (cell type LSTM) with 128 hidden layers
"""

class RNNController(Controller):

    def init_controller_params(self):
        self.rnn_dim = 300
        self.lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(self.rnn_dim)
        self.state = tf.Variable(tf.zeros([self.batch_size, self.rnn_dim]), trainable=False)
        self.output = tf.Variable(tf.zeros([self.batch_size, self.rnn_dim]), trainable=False)

    def nn_step(self, X, state):
        X = tf.convert_to_tensor(X)
        return self.lstm_cell(X, state)

    def update_state(self, update):
        return tf.group(self.output.assign(update[0]), self.state.assign(update[1]))

    def get_state(self):
        return LSTMStateTuple(self.output, self.state)
