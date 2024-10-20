import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import nibabel as nib
import pandas as pd
import tempfile
import cv2
import skimage.morphology as sk
import scipy.ndimage as sc
from focal_loss import BinaryFocalLoss
from sklearn.model_selection import train_test_split
from models.nifty_net import nifty_net

# +
HU_min = -500.
HU_max = 500.

HU_min2 = 0.
HU_max2 = 200.

# +
def normalize(image):
    image = (image - HU_min) / (HU_max - HU_min)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

def normalize2(image):
    image = (image - HU_min2) / (HU_max2 - HU_min2)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


m_ich = tf.keras.models.load_model('models/ICH.h5',compile=False)
m_ivh = tf.keras.models.load_model('models/IVH.h5',compile=False)
m_sah = tf.keras.models.load_model('models/SAH.h5',compile=False)
m_sah2 = nifty_net('models/niftynet/')


def predict(volume):
    pred1 = m_ich.predict(normalize(volume),verbose=1)
    pred2 = m_ivh.predict(normalize2(volume),verbose=1)
    pred3 = m_sah.predict(normalize(volume),verbose=1)
    pred4 = m_sah2.predict(volume)
    
    x = np.concatenate((normalize(volume),pred1,pred2,pred3,pred4),axis=3)
    return x
