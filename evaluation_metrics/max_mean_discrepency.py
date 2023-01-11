# MMD functions implemented in tensorflow: https://github.com/dougalsutherland/opt-mmd/blob/master/gan/mmd.py

import tensorflow as tf
import numpy as np


def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1.0] * sigmas.get_shape()[0]

    XX = tf.tensordot(X, X, axes=[[1, 2], [1, 2]])
    XY = tf.tensordot(X, Y, axes=[[1, 2], [1, 2]])
    YY = tf.tensordot(Y, Y, axes=[[1, 2], [1, 2]])

    X_sqnorms = tf.diag_part(XX)
    Y_sqnorms = tf.diag_part(YY)

    r = lambda x: tf.expand_dims(x, 0)
    c = lambda x: tf.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(tf.unstack(sigmas, axis=0), wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XX += wt * tf.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * tf.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * tf.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    return K_XX, K_XY, K_YY, tf.reduce_sum(wts)

def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = tf.cast(K_XX.get_shape()[0], tf.float32)
    n = tf.cast(K_YY.get_shape()[0], tf.float32)

    if biased:
        mmd2 = (tf.reduce_sum(K_XX) / (m * m)
                + tf.reduce_sum(K_YY) / (n * n)
                - 2 * tf.reduce_sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = tf.trace(K_XX)
            trace_Y = tf.trace(K_YY)

        mmd2 = ((tf.reduce_sum(K_XX) - trace_X) / (m * (m - 1))
                + (tf.reduce_sum(K_YY) - trace_Y) / (n * (n - 1))
                - 2 * tf.reduce_sum(K_XY) / (m * n))

    return mmd2

def mix_rbf_mmd2(X, Y, sigmas=(1, ), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def max_mean_discrepency(real_data, syn_data):

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    mmd2_value = sess.run(
        mix_rbf_mmd2(np.float32(real_data), np.float32(syn_data),
                     sigmas=tf.convert_to_tensor(bandwidths)))

    return np.sqrt(mmd2_value)