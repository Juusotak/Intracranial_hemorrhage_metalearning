import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input

def ensemble_model():
    
    inputs = Input(shape=(512,512,5))
    conv1 = layers.Conv2D(filters = 3, kernel_size = 3, strides= 1, padding = 'same',dilation_rate= (1,1), activation='relu')(inputs)
    batch_norm1 = layers.BatchNormalization(axis = -1)(conv1)
    output = layers.Conv2D(filters = 3, kernel_size = 1, activation = 'sigmoid', padding = 'same')(batch_norm1)
    
    model = tf.keras.Model(inputs = inputs, outputs = output)
    
    return model 