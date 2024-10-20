import numpy as np

import tensorflow as tf
import os


import tempfile
import cv2
import scipy.ndimage as sc
from all_you_need.niftynet import nifty_net


Dropout = 0.1

f1 = 10
f2 = 20
f3 = 40
f4 = 80
f5 = 160
f6 = 320

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


# -

def dice_coeff(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

# +
def tversky(y_true, y_pred):
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos))
    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + beta*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    loss = 1 - tversky(y_true, y_pred)
    return loss

def focal_tversky_loss(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    loss = tf.math.pow((1-pt_1), gamma)
    return loss

def focal_bce_loss(y_true, y_pred):
    pt_1 = (1-tf.keras.losses.binary_crossentropy(y_true,y_pred))
    loss = tf.math.pow((1-pt_1), gamma)
    return loss

def focal_bce_tversky(y_true, y_pred):
    loss = focal_bce_loss(y_true, y_pred) + tversky_loss(y_true, y_pred)
    return loss





m_ich = tf.keras.models.load_model('all_you_need/Models/ICH.h5',compile=False)
m_ivh = tf.keras.models.load_model('all_you_need/Models/IVH.h5',compile=False)
m_sah = tf.keras.models.load_model('all_you_need/Models/SAH.h5',compile=False)
m_sah2 = nifty_net()
m_all = tf.keras.models.load_model('all_you_need/Models/ALL.h5',compile=False)




def predict(volume, ensemble_model_config = 'ensemble_ICH_IVH_SAH1_SAH2'):

    ensemble_model_config = ensemble_model_config.replace('.h5','')

    
    base_net_predictions = {}

    image = normalize(volume)
    

    if 'ICH' in ensemble_model_config.upper():
        ich = m_ich.predict(normalize(volume),verbose=0)
        base_net_predictions['ICH'] = ich

    if 'IVH' in ensemble_model_config.upper():
        ivh = m_ivh.predict(normalize2(volume),verbose=0)
        base_net_predictions['IVH'] = ivh

    if 'SAH1' in ensemble_model_config.upper():
        sah_1 = m_sah.predict(normalize(volume),verbose=0)
        base_net_predictions['SAH_1'] = sah_1

    if 'SAH2' in ensemble_model_config.upper():
        sah_2 = m_sah2.predict(volume)
        base_net_predictions['SAH_2'] = sah_2



    x = [image]

    if 'ICH'.upper() in ensemble_model_config.upper():
        x.append(ich)

    if 'IVH'.upper() in ensemble_model_config.upper():
        x.append(ivh)

    if 'SAH1'.upper() in ensemble_model_config.upper():
        x.append(sah_1)

    if 'SAH2'.upper() in ensemble_model_config.upper():
        x.append(sah_2)

    return np.concatenate(x, axis = -1),base_net_predictions
