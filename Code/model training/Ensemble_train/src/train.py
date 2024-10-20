import sys
import subprocess

subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])

from sklearn.model_selection import train_test_split

import numpy as np
import os
from os import listdir
from os.path import join
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import argparse

from ensemble import ensemble_model
from datagenerator import DataGenerator
from utils import dice_coeff_ich, dice_coeff_ivh, dice_coeff_sah #,dice_coeff
from utils import Dice_loss_multi_w2, Dice_loss_multi_w, custom_loss, tvesky_loss #, dice_loss

from losses import dice_coeff, dice_loss, tversky, tversky_loss, log_dice_loss


from metrics import sensitivity_ich, sensitivity_ivh, sensitivity_sah 
from metrics import specifisity_ich, specifisity_ivh, specifisity_sah

from Azure_callback import AzureMlKerasCallback


from azureml.core import Run


run = Run.get_context()

parser = argparse.ArgumentParser()

#Parse the arguments:
parser.add_argument('--dataset_input', type=str)
parser.add_argument('--learning_rate', type=float, default = 0.07)
parser.add_argument('--epochs', type=int, default = 100)
parser.add_argument('--batch_size', type=int, default = 16)
args = parser.parse_args()

dataset_input = args.dataset_input
learning_rate = round(args.learning_rate,6)
epochs = args.epochs
batch_size = args.batch_size

os.makedirs('./outputs', exist_ok = True)
filepath = f'./outputs/{run.display_name}.h5'

x = np.load(join(dataset_input,'all_img.npy'))
y = np.load(join(dataset_input,'all_seg.npy'))




print('learning rate    : ', learning_rate)
print('epochs           : ', epochs)
print('batch size       : ', batch_size)
print('x shape, y shape : ',x.shape, y.shape)

x, x_val, y, y_val = train_test_split(x, y, test_size = 0.20, random_state = 28)

train_gen = DataGenerator(x,y, batch_size)
valid_gen = DataGenerator(x_val, y_val, batch_size)


model = ensemble_model()


metrics=[dice_coeff,dice_coeff_ich, dice_coeff_ivh,dice_coeff_sah, sensitivity_ich, sensitivity_ivh, sensitivity_sah, specifisity_ich, specifisity_ivh, specifisity_sah]


model.compile(optimizer = Adam(learning_rate = learning_rate), 
              loss = dice_loss,
              metrics=metrics, 
              run_eagerly = True)



checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_loss', mode='min', verbose=1,save_weights_only= True, save_best_only=True)

azureml_cb = AzureMlKerasCallback(run)

model.fit(train_gen,epochs=epochs,callbacks=[checkpoint,azureml_cb],validation_data = valid_gen)

