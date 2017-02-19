# Differentiable Neural Computer
# inspired by (http://www.nature.com/nature/journal/v538/n7626/full/nature20101.html)
# some ideas taken from https://github.com/Mostafa-Samir/DNC-tensorflow
# Sam Greydanus. February 2017. MIT License.

import tensorflow as tf
import numpy as np

class Memory():
    def __init__(self, R, W, N, batch_size):
        '''
        Defines how the interface vector zeta interacts with the memory state of the DNC
        Parameters:
        ----------
        R: the number of DNC read heads
        W: the DNC "word length" (length of each DNC memory vector)
        N: the number of DNC word vectors (corresponds to memory size)
        batch_size: the number of batches
        '''

        self.R = R
        self.W = W
        self.N = N
        self.batch_size = batch_size

        # when we go from 2D indexes to a flat 1D vector, we need to reindex using these shifts
        ix_flat_shifts = tf.constant(np.cumsum([0] + [N] * (batch_size - 1)), dtype=tf.int32)
        self.ix_flat_shifts = tf.expand_dims(ix_flat_shifts, [1])

        # N x N identity matrix
        self.I = tf.eye(N)
        self.eps = 1e-6
        self.state_keys = ['M', 'u', 'p', 'L', 'w_w', 'w_r', 'r']

    def zero_state(self):
        '''
        Supplies the initial state of the DNC's memory vector

        Returns: Tuple(7)
            dnc_state: contains initial values for (M, u, p, L, w_w, w_r, r) respectively. According to the DNC paper:
                M: (batch_size, N, W) the memory vector
                u: (batch_size, N) the usage vector
                p: (batch_size, N) the precedence weighting (helps update L)
                L: (batch_size, N, N) the temporal linkage matrix (helps DNC remember what order things were written)
                w_w: (batch_size, N) the write weighting - says where the DNC wrote word last time step
                w_r: (batch_size, N, R )the read vector - says which word vectors the DNC accessed last time step
        '''
        return [
            tf.fill([self.batch_size, self.N, self.W], self.eps),  # M
            tf.zeros([self.batch_size, self.N, ]),                 # u
            tf.zeros([self.batch_size, self.N, ]),                 # p
            tf.zeros([self.batch_size, self.N, self.N]),           # L
            tf.fill([self.batch_size, self.N, ], self.eps),        # w_w
            tf.fill([self.batch_size, self.N, self.R], self.eps),  # w_r
            tf.fill([self.batch_size, self.W, self.R], self.eps),  # r
        ]

    def content_addressing(self, M, kappa, beta):
        '''
        Computes the probabilities that each word vector in memory was the target of a given key (see paper)
        '''
        norm_M = tf.nn.l2_normalize(M, 2)
        norm_kappa = tf.nn.l2_normalize(kappa, 1)
        similiarity = tf.matmul(norm_M, norm_kappa)

        return tf.nn.softmax(similiarity * tf.expand_dims(beta, 1), 1)

    def update_u(self, u, w_r, w_w, f):
        '''
        Computes the new usage vector. This tells the DNC which memory slots are being used and which are free (see paper)
        '''
        f = tf.expand_dims(f, 1) # need to match w_r dimensions
        psi = tf.reduce_prod(1 - w_r * f, 2) # psi tells us what usage to reserve
        next_u = (u + w_w - u * w_w) * psi # update u based on what we wrote last time
        return next_u

    def get_allocation(self, next_u):
        '''
        Computes the allocation vector. This tells the DNC where it COULD write its next memory (see paper)
        '''
        u_sorted, u_ix = tf.nn.top_k(-1 * next_u, self.N) # sort by descending usage
        u_sorted = -1 * u_sorted
        a_sorted = (1 - u_sorted) * tf.cumprod(u_sorted, axis=1, exclusive=True) # classic DNC cumprod

        # indexing wizardry to account for multiple batches
        ix_flat = u_ix + self.ix_flat_shifts
        ix_flat = tf.reshape(ix_flat, (-1,))
        flat_array = tf.TensorArray(tf.float32, self.batch_size * self.N)

        a_scattered = flat_array.scatter(ix_flat, tf.reshape(a_sorted, (-1,))) # undo the sort
        a = a_scattered.stack() # put back into a Tensor
        return tf.reshape(a, (self.batch_size, self.N))

    def update_w_w(self, c_w, a, g_w, g_a):
        '''
        Computes the new write weighting. This tells the DNC where (and if) it will write its next memory (see paper)
        '''
        c_w = tf.squeeze(c_w) # want c_w as a (batched) vector
        next_w_w = g_w * (g_a * a + (1 - g_a) * c_w) # apply the allocation and write gates
        return next_w_w

    def update_M(self, M, w_w, v, e):
        '''
        Computes the new memry matrix. This is where the DNC actually stores memories (see paper)
        '''
        # expand data to force matmul to behave as an outer product
        w_w = tf.expand_dims(w_w, 2)
        v = tf.expand_dims(v, 1)
        e = tf.expand_dims(e, 1)

        # think of the memory update as a bunch of elementwise interpolations
        M_erase = M * (1 - tf.matmul(w_w, e))
        M_write = tf.matmul(w_w, v)
        next_M = M_erase + M_write
        return next_M

    def update_p(self, p, w_w):
        '''
        Updates the precedence vector. This tells the DNC how much each location was the last one written to (see paper)
        '''
        interpolate = 1 - tf.reduce_sum(w_w, 1, keep_dims=True)
        next_p = interpolate * p + w_w
        return next_p

    def update_L(self, p, L, w_w):
        '''
        Updates the temoral linkage matrix. This tells the DNC what order it has written things to memory (see paper)
        '''
        w_w = tf.expand_dims(w_w, 2)
        p = tf.expand_dims(p, 1)

        # compute "outer sum" of w_w
        c_w_w = tf.reshape(w_w, (-1, self.N, 1))
        U = tf.tile(c_w_w,[1, 1, self.N])
        w_w_outer_sum = U + tf.transpose(U, [0, 2, 1])

        next_L = (1 - w_w_outer_sum) * L + tf.matmul(w_w, p) # update L
        return (1 - self.I) * next_L # get rid of links to self

    def get_bf_w(self, w_r, L):
        '''
        Gets the write locations immediately before and after a given write location. This lets the DNC traverse memories in order (see paper)
        '''
        f_w = tf.matmul(L, w_r)
        b_w = tf.matmul(L, w_r, adjoint_a=True) # transpose the first argument
        return f_w, b_w

    def update_w_r(self, c_w, f_w, b_w, pi):
        '''
        Updates the read weighting. This tells the DNC's read heads which memories to extract (see paper)
        '''
        backward = tf.expand_dims(pi[:, 0, :], 1) * b_w
        content = tf.expand_dims(pi[:, 1, :], 1) * c_w
        forward = tf.expand_dims(pi[:, 2, :], 1) * f_w
        next_w_r = backward + content + forward
        return next_w_r

    def update_r(self, M, w_r):
        '''
        Gets the DNC's output. This vector contains the outputs of the DNC's read heads (see paper)
        '''
        return tf.matmul(M, w_r, adjoint_a=True) # transpose the first argument

    def write(self, zeta, state):
        '''
        Performs a write action on the DNC's memory
        Parameters:
        ----------
        zeta: dict
            variable names (string) mapping to tensors (Tensor) includes:
                'kappa_r': (batch_size, W, R) read key (there are R of them)
                'beta_r': (batch_size, R) read strength
                'kappa_w': (batch_size, W, 1) write key
                'beta_w': (batch_size, 1) write strength
                'e': (batch_size, W) erase vector
                'v': (batch_size, W) write vector
                'f': (batch_size, R) free gates (R of them)
                'g_a': (batch_size, 1) allocation gate
                'g_w': (batch_size, 1) write gate
                'pi': (batch_size, 3, R) read modes (backward, content, forward)
                        ... see paper for more info
        state: dict
            contains initial values for (M, u, p, L, w_w, w_r, r) respectively. According to the DNC paper:
                M: (batch_size, N, W) the memory vector
                u: (batch_size, N) the usage vector
                p: (batch_size, N) the precedence weighting (helps update L)
                L: (batch_size, N, N) the temporal linkage matrix (helps DNC remember what order things were written)
                w_w: (batch_size, N) the write weighting - says where the DNC wrote word last time step
                w_r: (batch_size, N, R )the read vector - says which word vectors the DNC accessed last time step
        Returns: Tuple(5)
            next_u: Tensor
            next_w_w: Tensor
            next_M: Tensor
            next_L: Tensor
            next_pL Tensor
        '''
        c_w      = self.content_addressing(state['M'], zeta['kappa_w'], zeta['beta_w'])
        next_u   = self.update_u(state['u'], state['w_r'], state['w_w'], zeta['f'])

        a        = self.get_allocation(next_u)
        next_w_w = self.update_w_w(c_w, a, zeta['g_w'], zeta['g_a'])
        next_M   = self.update_M(state['M'], next_w_w, zeta['v'], zeta['e'])
        next_L   = self.update_L(state['p'], state['L'], next_w_w)
        next_p   = self.update_p(state['p'], next_w_w)

        return next_u, next_w_w, next_M, next_L, next_p

    def read(self, zeta, state):
        '''
        Performs a read action on the DNC's memory
        Parameters:
        ----------
        zeta: dict
            variable names (string) mapping to tensors (Tensor) includes:
                'kappa_r': (batch_size, W, R) read key (there are R of them)
                'beta_r': (batch_size, R) read strength
                'kappa_w': (batch_size, W, 1) write key
                'beta_w': (batch_size, 1) write strength
                'e': (batch_size, W) erase vector
                'v': (batch_size, W) write vector
                'f': (batch_size, R) free gates (R of them)
                'g_a': (batch_size, 1) allocation gate
                'g_w': (batch_size, 1) write gate
                'pi': (batch_size, 3, R) read modes (backward, content, forward)
                        ... see paper for more info
        state: dict
            contains initial values for (M, u, p, L, w_w, w_r, r) respectively. According to the DNC paper:
                M: (batch_size, N, W) the memory vector
                u: (batch_size, N) the usage vector
                p: (batch_size, N) the precedence weighting (helps update L)
                L: (batch_size, N, N) the temporal linkage matrix (helps DNC remember what order things were written)
                w_w: (batch_size, N) the write weighting - says where the DNC wrote word last time step
                w_r: (batch_size, N, R )the read vector - says which word vectors the DNC accessed last time step
        Returns: Tuple(2)
            next_w_r: Tensor
            next_r: Tensor
        '''
        c_w =      self.content_addressing(state['M'], zeta['kappa_r'], zeta['beta_r'])
        f_w, b_w = self.get_bf_w(state['w_r'], state['L'])
        next_w_r = self.update_w_r(c_w, f_w, b_w, zeta['pi'])
        next_r =   self.update_r(state['M'], next_w_r)
        return next_w_r, next_r

    def step(self, zeta, state):
        '''
        Combines the read and write operations into a single memory update step.
        '''
        state['u'], state['w_w'], state['M'], state['L'], state['p'] = self.write(zeta, state)
        state['w_r'], state['r'] = self.read(zeta, state)
        return state