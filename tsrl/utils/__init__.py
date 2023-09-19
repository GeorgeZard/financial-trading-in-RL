import numba as nb
import numpy as np


def create_decay(start, stop, total_steps):
    def decay(cur_step):
        p = np.clip(cur_step / total_steps, 0.0, 1.0)
        return (1 - p) * start + p * stop

    return decay


def create_hyperbolic_decay(start, stop, total_steps, hyperbolic_steps=10):
    max_hb = 1 / (1 + hyperbolic_steps)

    def decay(cur_step):
        progress_frac = (cur_step / total_steps)
        p = np.clip((1 / (1 + progress_frac * hyperbolic_steps)) - progress_frac * max_hb, 0.0, 1.0)
        return p * start + (1 - p) * stop

    return decay


def batched_iterator(*args, batch_size=None):
    assert batch_size is not None
    assert (all([len(ar) == len(args[0]) for ar in args]))
    length = len(args[0])
    i = 0
    for i in range(batch_size, length, batch_size):
        if len(args) > 1:
            yield [ar[i - batch_size:i] for ar in args]
        else:
            yield args[0][i - batch_size:i]
    if i < length:
        if len(args) > 1:
            yield [ar[i:length] for ar in args]
        else:
            yield args[0][i:length]


def random_batched_iterator(*args, batch_size=None, include_idxs=False, disable_randomness=False):
    assert batch_size is not None
    assert (all([len(ar) == len(args[0]) for ar in args])), "All inputs must have the same len"
    length = len(args[0])
    i = 0
    idxs = np.random.permutation(length)
    if disable_randomness:
        idxs = np.arange(length)
    for i in range(batch_size, length, batch_size):
        bidx = [] if not include_idxs else [idxs[i - batch_size:i]]
        if len(args) > 1 or include_idxs:
            yield [ar[idxs[i - batch_size:i]] for ar in args] + bidx
        else:
            yield args[0][idxs[i - batch_size:i]]
    if i < length:
        bidx = [] if not include_idxs else [idxs[i:length]]
        if len(args) > 1 or include_idxs:
            yield [ar[idxs[i:length]] for ar in args] + bidx
        else:
            yield args[0][idxs[i:length]]


@nb.njit
def fast_categorical(y_in, num_classes):
    y = np.array(y_in, dtype=np.int64)
    input_shape = y.shape
    y = y.ravel()
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    for i in range(n):
        categorical[i, y[i]] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


import math


class RunningScalarMean:
    def __init__(self):
        self.mean = 0.
        self.count = 1e-6
        self.n_bad_values = 0

    def reset(self):
        self.mean = 0.
        self.count = 1e-6
        self.n_bad_values = 0

    def update(self, x):
        if not math.isfinite(x):
            self.n_bad_values += 1
            return
        batch_count = 1
        delta = x - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        self.count = tot_count


class RunningMean:
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot_count
        self.count = tot_count


"""
This can be generalized to allow parallelization with AVX, with GPUs, and computer clusters, and to covariance 
https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
"""
class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


@nb.njit()
def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


import torch


def zero_nan_param_grads(parameters, zero_nans=True):
    nb_nans = 0
    for i, p in enumerate(parameters):
        vals = p.grad
        if vals is None:
            continue
        nan_idxs = ~torch.isfinite(vals)
        nb_nans += nan_idxs.sum()
        if nb_nans > 0 and zero_nans:
            p.grad[nan_idxs] = 0.
    return nb_nans


def zero_nans(x):
    x[torch.isnan(x)] = 0.
    x[torch.isinf(x)] = 0.
    return x
