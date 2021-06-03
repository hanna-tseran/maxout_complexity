from enum import Enum

import numpy as np

class InitMethod(Enum):
    HE_NORMAL = 1
    MAXOUT_HE_NORMAL = 2
    SPHERE = 3
    MANY_REGIONS = 4

def maxout_he_std(fan_in, K):
    if K == 1: # We use this std for a linear layer
        return np.sqrt(2. / (fan_in))
    if K == 2:
        return np.sqrt(1. / (fan_in))
    elif K == 3:
        return np.sqrt(2. * np.pi / ((np.sqrt(3) + 2 * np.pi) * fan_in))
    elif K == 4:
        return np.sqrt(np.pi / ((np.sqrt(3) + np.pi) * fan_in))
    elif K == 5:
        return np.sqrt(0.5555 / (fan_in))
    else:
        raise Exception('Wrong init!')
    return init_distr

def init_params(init, K, fan_in, fan_out, zero_bias):
    weight_shape = [fan_out, K, fan_in] if K > 1 else [fan_out, fan_in]
    bias_shape = [fan_out, K] if K > 1 else [fan_out]

    if init == InitMethod.HE_NORMAL:
        init_distr = lambda fan_in, size: np.random.normal(loc=0., scale=np.sqrt(2. / (fan_in)), size=size)
        weight = init_distr(fan_in=fan_in, size=weight_shape)

        if zero_bias:
            bias = np.zeros(bias_shape)
        else:
            bias = init_distr(fan_in=fan_in, size=bias_shape)

    elif init == InitMethod.MAXOUT_HE_NORMAL:
        init_distr = lambda fan_in, size: np.random.normal(loc=0., scale=maxout_he_std(fan_in=fan_in, K=K), size=size)
        weight = init_distr(fan_in=fan_in, size=weight_shape)

        if zero_bias:
            bias = np.zeros(bias_shape)
        else:
            bias = init_distr(fan_in=fan_in, size=bias_shape)

    elif init == InitMethod.SPHERE:
        init_distr = lambda fan_in, size: np.random.normal(loc=0., scale=maxout_he_std(fan_in=fan_in, K=K), size=size)
        weight = init_distr(fan_in=fan_in, size=weight_shape)
        bias = init_distr(fan_in=fan_in, size=bias_shape)
        for i in range(fan_out):
            for j in range(K):
                norm = np.linalg.norm(weight[i][j] + [bias[i][j]])
                weight[i][j] = weight[i][j] / norm
                bias[i][j] = bias[i][j] / norm
                c = 1 / np.sqrt(K * fan_in)
                bias[i][j] = np.abs(bias[i][j]) - c

    elif init == InitMethod.MANY_REGIONS:
        init_distr = lambda fan_in, size: np.random.normal(loc=0., scale=maxout_he_std(fan_in=fan_in, K=K), size=size)
        weight = np.zeros(weight_shape)
        bias = np.zeros(bias_shape)
        for i in range(fan_out):
            v = init_distr(fan_in=fan_in, size=[fan_in])
            for j in range(K):
                weight[i][j] = v * np.cos(np.pi * (j + 1) / K + 0.01 * np.random.normal(loc=0., scale=1.))
                bias[i][j] = np.sin(np.pi * (j + 1) / K + 0.01 * np.random.normal(loc=0., scale=1.))
    else:
        raise Exception(f'Wrong init {init}')
    return weight, bias

def get_init_std(init, K, fan_in):
    if init == InitMethod.HE_NORMAL:
        return np.sqrt(2. / (fan_in))
    elif init == InitMethod.MAXOUT_HE_NORMAL:
        return maxout_he_std(fan_in=fan_in, K=K)

    # This is a stub, to avoid an exception
    # Sphere init has no distribution and variance of the many ragions initialization is difficult to estimate
    return maxout_he_std(fan_in=fan_in, K=K)
