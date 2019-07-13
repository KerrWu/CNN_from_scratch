

import numpy as np


def batch_norm(x, gamma=1, beta=0, epsilon=2e-5):
    '''

    :param x: shape=[batch ,h, w, c]
    :return:
    '''

    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x = (x-mean)/(var+epsilon)
    x = gamma*x+beta

    return x
