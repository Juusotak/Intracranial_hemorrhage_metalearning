import tensorflow as tf
import numpy as np

smooth = 1e-8



def dice_coeff(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_coeff_ich(y_true, y_pred):

    y_true_f = tf.reshape(y_true[...,0:1], [-1])
    y_pred_f = tf.reshape(y_pred[...,0:1], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_coeff_ivh(y_true, y_pred):
    
    y_true_f = tf.reshape(y_true[...,1:2], [-1])
    y_pred_f = tf.reshape(y_pred[...,1:2], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_coeff_sah(y_true, y_pred):
    
    y_true_f = tf.reshape(y_true[...,2:3], [-1])
    y_pred_f = tf.reshape(y_pred[...,2:3], [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

import tensorflow as tf


class Dice_loss_multi_w(tf.keras.losses.Loss):
    def __init__(self, w_not = 1.25, w_bleed = 1.0):
        super().__init__()
        self.smooth = 1e-8
        
        self.w_not = w_not
        self.w_bleed = w_bleed
        
    def call(self, y_true, y_pred):
        
        
        ## y_true
        y_true_ich = y_true[...,0:1]
        y_true_not_ich = (y_true[...,0:1] -1)*-1
        
        y_true_ivh = y_true[...,1:2]
        y_true_not_ivh = (y_true[...,1:2]-1)*-1
        
        y_true_sah = y_true[...,2:3]
        y_true_not_sah = (y_true[...,2:3] -1)*-1


        ## prediction
        y_pred_ich = y_pred[...,0:1]
        y_pred_not_ich = (y_pred[...,0:1] -1)*-1
        
        y_pred_ivh = y_pred[...,1:2]
        y_pred_not_ivh = (y_pred[...,1:2] - 1)*-1
        
        y_pred_sah = y_pred[...,2:3]
        y_pred_not_sah = (y_pred[...,2:3] -1)*-1
        
        
        
        
        
        
        
        
        
        #intersections ICH
        int_ich     = y_true_ich     *  y_pred_ich
        int_not_ich = y_true_not_ich *  y_pred_not_ich
        
        
        #intersections IVH
        int_ivh     = y_true_ivh      *  y_pred_ivh
        int_not_ivh = y_true_not_ivh  *  y_pred_not_ivh
        
        
        #intersections SAH
        int_sah      = y_true_sah      * y_pred_sah
        int_not_sah  = y_true_not_sah  * y_pred_not_sah
        
        # ICH
        upper_ich = tf.reduce_sum(int_ich)*2*self.w_bleed + tf.reduce_sum(int_not_ich)*2 * self.w_not                                                                                                                                                                                                   
        lower_ich = (tf.reduce_sum(y_true_ich) + tf.reduce_sum(y_pred_ich))*self.w_bleed + (tf.reduce_sum(y_true_not_ich) + tf.reduce_sum(y_pred_not_ich))*self.w_not
        
        # ivh
        upper_ivh = tf.reduce_sum(int_ivh)*2*self.w_bleed + tf.reduce_sum(int_not_ivh)*2 * self.w_not                                                                                                                                                                                                   
        lower_ivh = (tf.reduce_sum(y_true_ivh) + tf.reduce_sum(y_pred_ivh))*self.w_bleed + (tf.reduce_sum(y_true_not_ivh) + tf.reduce_sum(y_pred_not_ivh))*self.w_not
        
        # sah
        upper_sah = tf.reduce_sum(int_sah)*2*self.w_bleed + tf.reduce_sum(int_not_sah)*2 * self.w_not                                                                                                                                                                                                   
        lower_sah = (tf.reduce_sum(y_true_sah) + tf.reduce_sum(y_pred_sah))*self.w_bleed + (tf.reduce_sum(y_true_not_sah) + tf.reduce_sum(y_pred_not_sah))*self.w_not
        
        
        
        
        score =  (upper_ich + upper_ivh + upper_sah + self.smooth) / (lower_ich + lower_ivh + lower_sah + self.smooth)
        
        return 1 - score

class Dice_loss_multi_w2(tf.keras.losses.Loss):
    def __init__(self, w_ivh = 0.3, w_sah = 0.45):
        super().__init__()
        self.smooth = 1e-8
        
        self.w_ivh = w_ivh
        self.w_sah = w_sah
        self.w_ich = 1 - self.w_ivh - self.w_sah
        
    def call(self, y_true, y_pred):
        
        
        ## y_true
        y_true_ich = y_true[...,0:1]        
        y_true_ivh = y_true[...,1:2]        
        y_true_sah = y_true[...,2:3]

        ## prediction
        y_pred_ich = y_pred[...,0:1]        
        y_pred_ivh = y_pred[...,1:2]
        y_pred_sah = y_pred[...,2:3]
        
        
        #intersections
        int_ich     = y_true_ich     *  y_pred_ich
        int_ivh     = y_true_ivh      *  y_pred_ivh
        int_sah      = y_true_sah      * y_pred_sah

        
        # ich
        score_ich = (tf.reduce_sum(int_ich)*2 + self.smooth) / (tf.reduce_sum(y_true_ich) + tf.reduce_sum(y_pred_ich) + self.smooth)        
        
        # ivh
        score_ivh = (tf.reduce_sum(int_ivh)*2 + self.smooth) / (tf.reduce_sum(y_true_ivh) + tf.reduce_sum(y_pred_ivh) + self.smooth)            
        
        # sah
        score_sah = (tf.reduce_sum(int_sah)*2 + self.smooth) / (tf.reduce_sum(y_true_sah) + tf.reduce_sum(y_pred_sah) + self.smooth)

        
        score = score_ich*self.w_ich + score_ivh*self.w_ivh + score_sah*self.w_sah
  
        
        return 1 - score

class custom_loss(tf.keras.losses.Loss):
    
    def __init__(self, alpha = 0.25):
        
        super().__init__()
        
        self.smooth = 1e-8
        self.alpha = alpha 
        
    def call(self, y_true, y_pred):
        
        
        tp = y_true * y_pred # intersection
        
        fn = y_true - tp
        fp = y_pred - tp
        
        
        score =  (tf.reduce_sum(tp) + self.smooth) / (tf.reduce_sum(fn)*(1.0 + self.alpha) + tf.reduce_sum(tp) + tf.reduce_sum(fp)*(1.0 - self.alpha) + self.smooth)
        
        
        
        return 1- score 


class tvesky_loss(tf.keras.losses.Loss):
    
    def __init__(self, alpha = 0.75):
        
        super().__init__()
        
        self.smooth = 1e-8
        self.alpha = alpha 
        self.beta  = 1 -alpha
        
    def call(self, y_true, y_pred):
        
        
        tp = y_true * y_pred # intersection
        
        fn = y_true - tp
        fp = y_pred - tp
        
        
        score =  (tf.reduce_sum(tp) + self.smooth) / ( tf.reduce_sum(fn) *self.alpha  + tf.reduce_sum(tp) + tf.reduce_sum(fp)*self.beta + self.smooth)
        
        
        
        return 1- score 
        
