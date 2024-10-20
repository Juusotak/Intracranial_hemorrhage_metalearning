import tensorflow as tf
from tensorflow.keras.utils import Sequence
import numpy as np

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.idx = np.arange(len(self.x)).astype('int')
        

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))
    


    def __getitem__(self, idx):
        batch_x = self.x[self.idx[idx * self.batch_size:(idx+1)*self.batch_size]]


        image = batch_x[...,[0]]
        ich =  batch_x[...,[1]]
        ivh = batch_x[...,[2]]
        sah_1 = batch_x[...,[3]] 
        sah_2 = batch_x[...,[4]] 
        
       
        out_x = np.concatenate((image, ich, ivh,sah_1, sah_2), axis = -1)
 
        y = self.y[self.idx[idx * self.batch_size:(idx+1)*self.batch_size]]
        return out_x.astype('float32'), y.astype('float32')



    def on_epoch_end(self):
        np.random.shuffle(self.idx)

