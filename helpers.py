import numpy as np


def rollwin(x, window, stride = 1):
    """
    Transforms a 1D-array into rolling window samples

    Uses fast numpy shape manipulations and performs in O(1) time.

    Args
        x (ndarray): input array
        window (int): window size
        stride (int): window step

    Returns
        ndarray: 2D-array of shape ``(x.size - window, window)``, 
            where rows as consequitive samples from the array ``x``

    Raises
        ValueError: if input array is shorted than ``window``
    """

    if x.size < window:
        raise ValueError("Array must containt at least %i values" % window)
    dsize   = x.dtype.itemsize
    strides = (stride*dsize, dsize)
    shape   = ((x.size - window)//stride + 1, window)
    return np.lib.stride_tricks.as_strided(x, strides = strides, shape = shape)


def ts2mat(x, lags, stride = 1):
    """
    Transforms an 1D-array into rolling window samples with provided lag values.lag

    Uses fast numpy shape manipulations and performs in O(1) time.

    Args
        x (ndarray): input array
        lags (list of int): lag values >= 0
        stride (int): window step

    Returns
        ndarray: 2D-array of shape ``(x.size - max(lags), lags.size)``, 
            where rows as consequitive samples from the array ``x``
    """

    lags = np.array(lags)
    window = np.max(lags) + 1
    idx = np.sort(window - lags - 1)
    X = rollwin(x, window, stride)
    return X[:,idx]


def tssplit(y, train_size):
    if isinstance(train_size, float):
        n = int(y.size * train_size)
    else:
        n = train_size
    return y[:n], y[n:]