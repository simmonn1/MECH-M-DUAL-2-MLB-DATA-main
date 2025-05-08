import numpy as np
import math
import pywt


def rescale(data, nb):
    """Helper function for wavelet scaling"""
    x = np.abs(data)
    x = x - np.min(x)
    x = nb * x / np.max(x)
    x = 1 + np.fix(x)
    x[x > nb] = nb
    return x


def transform(images: list) -> list:
    """Transform an array of images via the haar wavelet and scales the
    horzontal and vertical part."""
    l, w = images.shape
    images_w = np.zeros((l // 4, w))
    for i in range(w):
        A = np.reshape(images[:, i], (math.isqrt(l), math.isqrt(l)))
        [_, (cH1, cV1, _)] = pywt.wavedec2(A, wavelet="haar", level=1)
        images_w[:, i] = np.matrix.flatten(rescale(cH1, 256) +
                                           rescale(cV1, 256))
    return images_w
