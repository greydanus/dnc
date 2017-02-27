# Differentiable Neural Computer
# inspired by (http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html)
# some ideas taken from https://github.com/Mostafa-Samir/DNC-tensorflow
# Sam Greydanus. February 2017. MIT License.

import tensorflow as tf
import numpy as np
from memory import Memory

import os

class DNC:
    def __init__(self, make_controller, FLAGS, input_steps=None):
        '''
        Builds a TensorFlow graph for the Differentiable Neural Computer. Uses TensorArrays and a while loop for efficiency
        Parameters:
        ----------
        make_controller: Controller
            An object class which inherits from the Controller class. We build the object in this function
        FLAGS: a set of TensorFlow FlagValues which must include
            FLAGS.xlen: the length of the input vector of the controller
            FLAGS.ylen: the length of the output vector of the controller
            FLAGS.batch_size: the number of batches
            FLAGS.R: the number of DNC read heads
            FLAGS.W: the DNC "word length" (length of each DNC memory vector)
            FLAGS.N: the number of DNC word vectors (corresponds to memory size)
        '''
        self.xlen = xlen = FLAGS.xlen
        self.ylen = ylen = FLAGS.ylen
        self.batch_size = batch_size = FLAGS.batch_size
        self.R = R = FLAGS.R
        self.W = W = FLAGS.W
        self.N = N = FLAGS.N

        # create 1) the DNC's memory and 2) the DNC's controller
        self.memory = Memory(R, W, N, batch_size)
        self.controller = make_controller(FLAGS)

        # input data placeholders
        self.X = tf.placeholder(tf.float32, [batch_size, None, xlen], name='X')
        self.y = tf.placeholder(tf.float32, [batch_size, None, ylen], name='y')
        self.tsteps = tf.placeholder(tf.int32, name='tsteps')
        self.input_steps = input_steps if input_steps is not None else self.tsteps

        self.X_tensor_array = self.unstack_time_dim(self.X)

        # initialize states
        self.nn_state = self.controller.get_state()
        self.dnc_state = self.memory.zero_state()

        # values for which we want a history
        self.hist_keys = ['y_hat', 'f', 'g_a', 'g_w', 'w_r', 'w_w', 'u']
        dnc_hist = [tf.TensorArray(tf.float32, self.tsteps, clear_after_read=False) for _ in range(len(self.hist_keys))]

        # loop through time
        with tf.variable_scope("dnc_scope", reuse=True) as scope:
            time = tf.constant(0, dtype=tf.int32)

            output = tf.while_loop(
                cond=lambda time, *_: time < self.tsteps,
                body=self.step,
                loop_vars=(time, self.nn_state, self.dnc_state, dnc_hist),
                )
        (_, self.next_nn_state, self.next_dnc_state, dnc_hist) = output

        # write down the history
        controller_dependencies = [self.controller.update_state(self.next_nn_state)]
        with tf.control_dependencies(controller_dependencies):
            self.dnc_hist = {self.hist_keys[i]: self.stack_time_dim(v) for i, v in enumerate(dnc_hist)} # convert to dict

    def step2(self, time, nn_state, dnc_state, dnc_hist):

        # map from tuple to dict for readability
        dnc_state = {self.memory.state_keys[i]: v for i, v in enumerate(dnc_state)}
        dnc_hist = {self.hist_keys[i]: v for i, v in enumerate(dnc_hist)}

        # one full pass!
        X_t = self.X_tensor_array.read(time)
        v, zeta, next_nn_state = self.controller.step(X_t, dnc_state['r'], nn_state)
        next_dnc_state = self.memory.step(zeta, dnc_state)
        y_hat = self.controller.next_y_hat(v, next_dnc_state['r'])

        dnc_hist['y_hat'] = dnc_hist['y_hat'].write(time, y_hat)
        dnc_hist['f']   = dnc_hist['f'].write(time, zeta['f'])
        dnc_hist['g_a'] = dnc_hist['g_a'].write(time, zeta['g_a'])
        dnc_hist['g_w'] = dnc_hist['g_w'].write(time, zeta['g_w'])
        dnc_hist['w_r'] = dnc_hist['w_r'].write(time, next_dnc_state['w_r'])
        dnc_hist['w_w'] = dnc_hist['w_w'].write(time, next_dnc_state['w_w'])
        dnc_hist['u']   = dnc_hist['u'].write(time, next_dnc_state['u'])

        # map from dict to tuple for tf.while_loop :/
        next_dnc_state = [next_dnc_state[k] for k in self.memory.state_keys]
        dnc_hist = [dnc_hist[k] for k in self.hist_keys]

        time += 1
        return time, next_nn_state, next_dnc_state, dnc_hist

    def step(self, time, nn_state, dnc_state, dnc_hist):
        '''
        Performs the feedforward step of the DNC in order to get the DNC output
        Parameters:
        ----------
        time: Constant 1-D Tensor
            the current time step of the model
        nn_state: LSTMStateTensor or another type of state tensor
            for the controller network
        dnc_state: Tuple
            set of 7 Tensors which define the current state of the DNC (M, u, p, L, w_w, w_r, r) ...see paper
        dnc_hist: Tuple
            set of 7 TensorArrays which track the historical states of the DNC (y_hat, f, g_a, g_w, w_r, w_w, u). Good for visualization
        Returns: Tuple
            same as input parameters, but updated for the current time step
        '''

        # map from tuple to dict for readability
        dnc_state = {self.memory.state_keys[i]: v for i, v in enumerate(dnc_state)}
        dnc_hist = {self.hist_keys[i]: v for i, v in enumerate(dnc_hist)}

        def use_prev_output():
            y_prev = tf.concat((dnc_hist['y_hat'].read(time-1), tf.zeros([self.batch_size, self.xlen - self.ylen])), axis=1)
            return tf.reshape(y_prev, (self.batch_size, self.xlen))

        def use_input_array():
            return self.X_tensor_array.read(time)

        # one full pass!
        X_t = tf.cond(time < self.input_steps, use_input_array, use_prev_output)
        v, zeta, next_nn_state = self.controller.step(X_t, dnc_state['r'], nn_state)
        next_dnc_state = self.memory.step(zeta, dnc_state)
        y_hat = self.controller.next_y_hat(v, next_dnc_state['r'])

        dnc_hist['y_hat'] = dnc_hist['y_hat'].write(time, y_hat)
        dnc_hist['f']   = dnc_hist['f'].write(time, zeta['f'])
        dnc_hist['g_a'] = dnc_hist['g_a'].write(time, zeta['g_a'])
        dnc_hist['g_w'] = dnc_hist['g_w'].write(time, zeta['g_w'])
        dnc_hist['w_r'] = dnc_hist['w_r'].write(time, next_dnc_state['w_r'])
        dnc_hist['w_w'] = dnc_hist['w_w'].write(time, next_dnc_state['w_w'])
        dnc_hist['u']   = dnc_hist['u'].write(time, next_dnc_state['u'])

        # map from dict to tuple for tf.while_loop :/
        next_dnc_state = [next_dnc_state[k] for k in self.memory.state_keys]
        dnc_hist = [dnc_hist[k] for k in self.hist_keys]

        time += 1
        return time, next_nn_state, next_dnc_state, dnc_hist

    def get_outputs(self):
        '''
        Allows user to access the output of the DNC after all time steps have been executed
        Returns: tuple
            y_hat: Tensor (batch_size, controller_dim)
                The DNC's ouput
            dnc_hist: Tuple
                Set of Tensors which contain values of (y_hat, f, g_a, g_w, w_r, w_w, u) respectively for all time steps
        '''
        return self.dnc_hist['y_hat'], self.dnc_hist

    def stack_time_dim(self, v):
        '''
        Stacks a TensorArray along its time dimension, then transposes so that the time dimension is at index [1]
        Parameters:
        ----------
        v: TensorArray [(batch_size, ...), ...]
            An array of n-dimensional tensor where for each, the first dimension is the batch dimension
        Returns: Tensor (batch_size, ylen)
            u: Tensor (batch_size, tsteps, ...)
                The stacked tensor with index [1] as the time dimension
        '''
        stacked = v.stack()
        return tf.transpose(stacked, [1,0] + range(2, len(stacked.get_shape())) )

    def unstack_time_dim(self, v):
        '''
        Unstacks a TensorArray along its time dimension
        Parameters:
        ----------
        v: Tensor (batch_size, tsteps, ...)
            An n-dimensional tensor where dim[0] is the batch dimension and dim[1] is the time dimension
        Returns: TensorArray [(batch_size, ...) ...]
            u: Tensor (batch_size, tsteps, ...)
                An array of n-dimensional tensor where, for each, the first dimension is the batch dimension
        '''
        array = tf.TensorArray(dtype=v.dtype, size=self.tsteps)
        make_time_dim_first = [1, 0] + range(2, len(v.get_shape()))
        v_T = tf.transpose(v, make_time_dim_first)
        return array.unstack(v_T)
