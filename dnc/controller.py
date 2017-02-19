# Differentiable Neural Computer
# inspired by (http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html)
# some ideas taken from https://github.com/Mostafa-Samir/DNC-tensorflow
# Sam Greydanus. February 2017. MIT License.

import tensorflow as tf
import numpy as np

class Controller():
    def __init__(self, FLAGS):
        '''
        An interface that defines how the neural network "controller" interacts with the DNC
        Parameters:
        ----------
        FLAGS: a set of TensorFlow FlagValues which must include
            FLAGS.xlen: the length of the input vector of the controller
            FLAGS.ylen: the length of the output vector of the controller
            FLAGS.batch_size: the number of batches
            FLAGS.R: the number of DNC read heads
            FLAGS.W: the DNC "word length" (length of each DNC memory vector)

        Returns: Tensor (batch_size, nn_output_size)
        '''

        self.xlen = FLAGS.xlen
        self.ylen = FLAGS.ylen
        self.batch_size = FLAGS.batch_size
        self.R = R = FLAGS.R
        self.W = W = FLAGS.W

        self.chi_dim = self.xlen + self.W * self.R
        self.zeta_dim = W*R + 3*W + 5*R + 3
        self.r_dim = W*R

        # define network vars
        self.params = {}
        with tf.name_scope("controller"):
            self.init_controller_params()
            self.controller_dim = self.get_controller_dim()

            init = tf.truncated_normal_initializer(stddev=0.075, dtype=tf.float32)
            self.params['W_z'] = tf.get_variable("W_z", [self.controller_dim, self.zeta_dim], initializer=init)
            self.params['W_v'] = tf.get_variable("W_v", [self.controller_dim, self.ylen], initializer=init)
            self.params['W_r'] = tf.get_variable("W_r", [self.r_dim, self.ylen], initializer=init)

    def init_controller_params(self):
        '''
        Initializes all the parameters of the neural network controller
        '''
        raise NotImplementedError("init_controller_params does not exist")

    def nn_step(self, chi, state):
        '''
        Performs the feedforward step of the controller in order to get the DNC interface vector, zeta
        Parameters:
        ----------
        chi: Tensor (batch_size, chi_dim)
            the input concatenated with the previous output of the DNC
        state: LSTMStateTensor or another type of state tensor
        Returns: Tuple
            zeta_hat: Tensor (batch_size, controller_dim)
            next_state: LSTMStateTensor or another type of state tensor
        '''
        raise NotImplementedError("nn_step does not exist")

    def zero_state(self):
        '''
        Returns the initial state of the controller. If the controller is not recurrent, it still needs to return a dummy value
        Returns: LSTMStateTensor or another type of state tensor
            nn_state: LSTMStateTensor or another type of state tensor
        '''
        raise NotImplementedError("get_state does not exist")

    def get_controller_dim(self):
        '''
        Feeds zeros through the controller and obtains an output in order to find the controller's output dimension
        Returns: int
            controller_dim: the output dimension of the controller
        '''
        test_chi = tf.zeros([self.batch_size, self.chi_dim])
        nn_output, nn_state = self.nn_step(test_chi, state=None)
        return nn_output.get_shape().as_list()[-1]

    def prepare_interface(self, zeta_hat):
        '''
        Packages the interface vector, zeta, as a dictionary of variables as described in the DNC Nature paper
        Parameters:
        ----------
        zeta_hat: Tensor (batch_size, zeta_dim)
            the interface vector before processing, zeta_hat
        Returns: dict
            zeta: variable names (string) mapping to tensors (Tensor)
        '''
        zeta = {}
        R, W = self.R, self.W
        splits = np.cumsum([0,W*R,R,W,1,W,W,R,1,1,3*R])
        vs = [zeta_hat[:, splits[i]:splits[i+1]] for i in range(len(splits)-1)]

        kappa_r = tf.reshape(vs[0], (-1, W, R))
        beta_r = tf.reshape(vs[1], (-1, R))
        kappa_w = tf.reshape(vs[2], (-1, W, 1))
        beta_w = tf.reshape(vs[3], (-1, 1))
        e = tf.reshape(vs[4], (-1, W))
        v = tf.reshape(vs[5], (-1, W))
        f = tf.reshape(vs[6], (-1, R))
        g_a = vs[7]
        g_w = vs[8]
        pi = tf.reshape(vs[9], (-1, 3, R))

        zeta['kappa_r'] = kappa_r
        zeta['beta_r'] = 1 + tf.nn.softplus(beta_r)
        zeta['kappa_w'] = kappa_w
        zeta['beta_w'] = 1 + tf.nn.softplus(beta_w)
        zeta['e'] = tf.nn.sigmoid(e)
        zeta['v'] = v
        zeta['f'] = tf.nn.sigmoid(f)
        zeta['g_a'] = tf.nn.sigmoid(g_a)
        zeta['g_w'] = tf.nn.sigmoid(g_w)
        zeta['pi'] = tf.nn.softmax(pi, 1)

        return zeta

    def step(self, X, r_prev, state):
        '''
        The sum of operations executed by the Controller at a given time step before interfacing with the DNC
        Parameters:
        ----------
        X: Tensor (batch_size, chi_dim)
            the input for this time step
        r_prev: previous output of the DNC
        state: LSTMStateTensor or another type of state tensor
        Returns: Tuple
            v: Tensor (batch_size, ylen)
                The controller's output (eventually added elementwise to the DNC output)
            zeta: The processed interface vector which the network will use to interact with the DNC
            nn_state: LSTMStateTensor or another type of state tensor
        '''
        r_prev = tf.reshape(r_prev, (-1, self.r_dim)) # flatten
        chi = tf.concat([X, r_prev], 1)
        nn_output, nn_state = self.nn_step(chi, state)

        v = tf.matmul(nn_output, self.params['W_v'])
        zeta_hat = tf.matmul(nn_output, self.params['W_z'])
        zeta = self.prepare_interface(zeta_hat)
        return v, zeta, nn_state

    def next_y_hat(self, v, r):
        '''
        The sum of operations executed by the Controller at a given time step after interacting with the DNC
        Parameters:
        ----------
        v: Tensor (batch_size, ylen)
            The controller's output (added elementwise to the DNC output)
        r_prev: the current output of the DNC
        Returns: Tensor (batch_size, ylen)
            y_hat: Tensor (batch_size, controller_dim)
                The DNC's ouput
        '''
        r = tf.reshape(r, (-1, self.W * self.R)) # flatten
        y_hat = v + tf.matmul(r, self.params['W_r'])
        return y_hat
