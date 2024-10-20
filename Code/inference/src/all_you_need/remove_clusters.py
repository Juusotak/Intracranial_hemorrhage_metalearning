import numpy as np
from scipy import ndimage


def remove(pred,n):


    out = np.zeros_like(pred)
    pos_sl = np.where(pred.reshape(pred.shape[0],-1).max(-1) != 0)[0]

    for sl in pos_sl:
        for ch in range(pred.shape[-1]):

            labeled, _ = ndimage.label(pred[sl,...,ch], structure = np.ones((3,3)))

            values, counts = np.unique(labeled, return_counts = True)
            values, counts = values[1:], counts[1:]

            for value, count in zip(values, counts):
                if count > n:
                    out[sl,...,ch] += labeled == value

    return out


    