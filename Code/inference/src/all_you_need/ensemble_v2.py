# +
from all_you_need import combo_v2
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
import numpy as np
import os


class Ensemble:

    def __init__(self,ensemble_model = 'ensemble_ich_ivh_sah1_sah2.h5'):

        self.ensemble_config = ensemble_model

        self.model = tf.keras.models.load_model(f'all_you_need/Ensemble_weights/{self.ensemble_config}', compile= False)
    
    def predict(self, image):
        prepare, base_predictions = combo_v2.predict(image,ensemble_model_config = self.ensemble_config)
        prediction = self.model.predict(prepare, verbose = 0)

        return prediction, base_predictions

