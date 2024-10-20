import tensorflow as tf

def sensitivity_ich(y_true, y_pred):
    smooth = 1e-8
    y_pred = y_pred[...,0:1]
    y_true = y_true[...,0:1]
    tp = y_true * y_pred
    fn = y_true - tp
    metric =   tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fn) + smooth) * 100
    
    return metric

def sensitivity_ivh(y_true, y_pred):
    smooth = 1e-8
    y_pred = y_pred[...,1:2]
    y_true = y_true[...,1:2]
    tp = y_true * y_pred
    fn = y_true - tp    
    metric =   tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fn) + smooth) * 100
    
    return metric

def sensitivity_sah(y_true, y_pred):
    smooth = 1e-8
    y_pred = y_pred[...,2:3]
    y_true = y_true[...,2:3]
    tp = y_true * y_pred
    fn = y_true - tp    
    metric =   tf.reduce_sum(tp) / (tf.reduce_sum(tp) + tf.reduce_sum(fn) + smooth) * 100
    
    return metric

def specifisity_ich(y_true, y_pred):
    smooth = 1e-8
    y_pred = y_pred[...,0:1]
    y_true = y_true[...,0:1]
    tp = y_true * y_pred
    fp = y_pred - tp
    tn = (1 - y_true)*(1- y_pred)   
    metric =   tf.reduce_sum(tn) / (tf.reduce_sum(tn) + tf.reduce_sum(fp) + smooth) * 100
    
    return metric

def specifisity_ivh(y_true, y_pred):
    smooth = 1e-8
    y_pred = y_pred[...,1:2]
    y_true = y_true[...,1:2]
    tp = y_true * y_pred
    fp = y_pred - tp
    tn = (1 - y_true)*(1- y_pred)   
    metric =   tf.reduce_sum(tn) / (tf.reduce_sum(tn) + tf.reduce_sum(fp) + smooth) * 100
    
    return metric

def specifisity_sah(y_true, y_pred):
    smooth = 1e-8
    y_pred = y_pred[...,2:3]
    y_true = y_true[...,2:3]
    tp = y_true * y_pred
    fp = y_pred - tp
    tn = (1 - y_true)*(1- y_pred) 
    metric =   tf.reduce_sum(tn) / (tf.reduce_sum(tn) + tf.reduce_sum(fp) + smooth) * 100
    
    return metric     