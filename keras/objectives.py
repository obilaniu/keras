from __future__ import absolute_import

from . import backend as K
from .utils.generic_utils import get_from_module


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def mean_absolute_percentage_error(y_true, y_pred):
    diff = K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                            K.epsilon(),
                                            None))
    return 100. * K.mean(diff, axis=-1)


def mean_squared_logarithmic_error(y_true, y_pred):
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.mean(K.square(first_log - second_log), axis=-1)


def squared_hinge(y_true, y_pred):
    return K.mean(K.square(K.maximum(1. - y_true * y_pred, 0.)), axis=-1)


def hinge(y_true, y_pred):
    return K.mean(K.maximum(1. - y_true * y_pred, 0.), axis=-1)


def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true)


def unnormalized_categorical_crossentropy(y_true, y_pred):
    """
    Implements in a numerically-stable manner a combined softmax and cross-
    entropy. Accepts unnormalized predictions y_pred and a probability
    distribution y_true. y_pred and y_true must be of the same shape and at
    least 2D; the last dimension must be the probability distribution and all
    others are batch dimensions.

    CE                   = -sum(log(softmax(y_pred)) * y_true, axis="all ~0")
    softmax(y_pred)      = exp(y_pred) / sum(exp(y_pred), axis=-1)
    log(softmax(y_pred)) = log(exp(y_pred) / sum(exp(y_pred), axis=-1))
                         = y_pred - log(sum(exp(y_pred), axis=-1))

    The input y_pred must be stabilized. This can be done by shifting it such
    that its maximum value is 0 and minimum is some low value (e.g. 30% of
    -log(MAX_VAL)).
    """

    # Floating-point preparations
    import numpy as np
    y       = y_pred                                  # Alias
    fMaxVal = np.finfo(y.dtype).max                   # Get dtype's MAXVAL
    fMinVal = np.finfo(y.dtype).min                   # Get dtype's MINVAL
    thresh  = -0.3 * np.log(fMaxVal)                  # Set a threshold at 30% of
                                                      # underflow value for exp().

    # Stabilization of y_pred
    y       = K.maximum(y, fMinVal)                   # NaNs and -Inf -> MINVAL
    y       = K.minimum(y, fMaxVal)                   # +Inf          -> MAXVAL
    y      -= K.max    (y, axis=-1, keepdims=True)    # Stabilize y s.t. max(y) == 0
    y       = K.maximum(y,  thresh)                   # Prevent underflow by clamping.

    # Numerically stable logsoftmax
    expy    = K.exp    (y)                            # Exponential of y
    sumexpy = K.sum    (expy, axis=-1, keepdims=True) # Sum of exponentials
    y      -= K.log    (sumexpy)                      # Logsoftmax.

    # Cross-entropy
    return -K.sum(y*y_true, axis=range(1, y.ndim))    # Sum out all axes ~0.


def sparse_categorical_crossentropy(y_true, y_pred):
    return K.sparse_categorical_crossentropy(y_pred, y_true)


def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)


def kullback_leibler_divergence(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1)
    y_pred = K.clip(y_pred, K.epsilon(), 1)
    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)


def poisson(y_true, y_pred):
    return K.mean(y_pred - y_true * K.log(y_pred + K.epsilon()), axis=-1)


def cosine_proximity(y_true, y_pred):
    y_true = K.l2_normalize(y_true, axis=-1)
    y_pred = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(y_true * y_pred, axis=-1)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


def get(identifier):
    return get_from_module(identifier, globals(), 'objective')
