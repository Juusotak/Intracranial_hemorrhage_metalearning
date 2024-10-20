class Logger:
    def __init__(self):
        self.log_params = {}
    def update(self,name,value):
        self.log_params[name] = value

    def print(self):
        print(self.log_params)
    
    def get_logs(self, log, name):

        if name not in self.log_params.keys():
            log[name] = None
        else:
            log[name] = self.log_params[name]

        return log





import os
from   os import listdir
from   os.path import join

import warnings
warnings.filterwarnings("ignore")

os.system('pip install opencv-python-headless')
os.system('pip install nilearn')
os.system('pip install nibabel')
os.system('pip install pydicom')
os.system('pip install odfpy')


import tensorflow as tf
import numpy as np
import csv
import nibabel as nib
import gc
import pandas as pd
import cv2
import time
import argparse


from all_you_need import ensemble_v2
from all_you_need import postprocess, testtime, remove_clusters
from all_you_need.utils import check_prediction, read_dicom_2
from all_you_need.resize_scan import resize_scan
from all_you_need.testtime import TTA_function

affine = np.array([[1,0,0,0],
                  [0,-1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]).astype(np.float64)

from azureml.core import Run


run = Run.get_context()

#Parse the arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_input', type=str)
parser.add_argument('--idx', type=str)
parser.add_argument('--gt_input', type=str)
parser.add_argument('--gt_file_name', type = str)
parser.add_argument('--ensemble_model', type = str)
parser.add_argument('--voting_coef',type=float, default=2.5)


args = parser.parse_args()
dataset_input = args.dataset_input
gt_data_input = args.gt_input
gt_file_name  = args.gt_file_name


ensemble_model = args.ensemble_model
voting_coef = args.voting_coef


idx = args.idx
idx = [int(x) for x in idx.split(',')]






gt_data = pd.read_excel(join(gt_data_input,gt_file_name))
gt_data = gt_data.loc[idx]
gt_data.set_index('AC-numero', inplace = True)



folder_name = 'user_logs'
os.makedirs(folder_name, exist_ok=True)

for x in ['tp','fn','fp']:
    os.makedirs(f'{folder_name}/{x}')


tp_tot = 0
fn_tot = 0
tn_tot = 0
fp_tot = 0
n_failed = 0

model_ensemble = ensemble_v2.Ensemble(ensemble_model = ensemble_model)

for ac, items in gt_data.iterrows():
    start_time = time.time()

    logger = Logger()

    tp = 0
    fn = 0
    tn = 0
    fp = 0

    log = {}


    ich = int(items.ICH == 1)
    ivh = int(items.IVH == 1)
    sah = int(items.SAV == 1)

    bleed = int( 1 in [ich,ivh,sah])


    gt = np.array([bleed,ich,ivh,sah,0,0])

    if gt[0] == 0:
        gt = 'neg'
    else:
        gt = 'bleed'

    name = ac
    
    logger.update('name', name)
    logger.update('bleed', bleed)
    logger.update('ich', ich)
    logger.update('ivh', ivh)
    logger.update('sah', sah)

    

    baseline = '_base_line' in items.current_folder
    follow_up = '_follow_up' in items.current_folder

    folder = items.current_folder.replace('_base_line','').replace('_follow_up','')
    src_folder = [x for x in os.listdir(dataset_input) if folder in x][0]

    if baseline:
        src_folder = join(src_folder, 'base_line')
    elif follow_up:
        src_folder = join(src_folder, 'follow_up')

    image = read_dicom_2(join(dataset_input, src_folder, ac, f'Image_data_{ac}'))
    image = image.astype('float32')


    image = resize_scan(image)
    thresh = 0.5
    pred, base_preds = model_ensemble.predict(image)
    pred = pred  > thresh
    pred = postprocess.postprocess(pred)
    pred = pred.max(-1, keepdims = True) # merge labels channels as one. 
    pred_ensemble = remove_clusters.remove(pred.copy(), n= 10)


    base_seg = 0
    for _, item in base_preds.items():
        base_seg += (item > thresh)
    base_seg = base_seg > thresh

    sum_pred = 0
    for _, item in base_preds.items(): 
        sum_pred += postprocess.postprocess_single_channel(item, thresh=thresh)
    sum_pred = sum_pred >= voting_coef
    
    over_lap = postprocess.compare_base_ensemble(pred, sum_pred)

    if over_lap:
        pred = remove_clusters.remove(pred,n=10)

    negative_ensemble     = np.count_nonzero(pred) == 0
    pos_pred_pixels_ensemble     = np.count_nonzero(pred)
    status_after_ensemble = check_prediction(gt, pred)

    logger.update('negative_after_ensemble', negative_ensemble)
    logger.update('pos_pred_pixels_ensemble', pos_pred_pixels_ensemble)
    logger.update('status_after_ensemble', status_after_ensemble)
    logger.update('over_lap', over_lap)

    
    if np.count_nonzero(pred):

        if not over_lap:
            TTA_used = 1
            before = np.count_nonzero(pred)
            
            pred = TTA_function(image, pred, model = model_ensemble)
            pred_TTA = pred.copy()

            negative_TTA       = np.count_nonzero(pred) == 0
            removed_pixels_TTA = before - np.count_nonzero(pred) 
            status_after_TTA = check_prediction(gt, pred)

        else:
            TTA_used           = 0
            negative_TTA       = None
            removed_pixels_TTA = None
            status_after_TTA   = None
            pred_TTA = None
            
        
        if np.count_nonzero(pred):
            
            before = np.count_nonzero(pred)

            pred = postprocess.filter_basesegmentation_ensemble(pred, base_seg)
            pred = remove_clusters.remove(pred, n = 125)
            

            filtering_used          = 1
            negative_filtering      = np.count_nonzero(pred) == 0
            pixel_removed_filtering = before - np.count_nonzero(pred)
            status_after_filtering  = check_prediction(gt, pred)

        else: 
            filtering_used          = 0
            negative_filtering      = None
            pixel_removed_filtering = None
            status_after_filtering  = None

        negative_after_post_processing = np.count_nonzero(pred) == 0

    else:
        TTA_used           = 0
        negative_TTA       = None
        removed_pixels_TTA = None
        status_after_TTA   = None
        pred_TTA           = None
        filtering_used          = 0
        negative_filtering      = None
        pixel_removed_filtering = None
        status_after_filtering  = None
        negative_after_post_processing = None

    logger.update('TTA_used', TTA_used)
    logger.update('negative_TTA', negative_TTA)
    logger.update('removed_pixels_TTA', removed_pixels_TTA)
    logger.update('status_after_TTA', status_after_TTA)
    logger.update('status_after_TTA', filtering_used)
    logger.update('negative_filtering', negative_filtering)
    logger.update('pixel_removed_filtering', pixel_removed_filtering)
    logger.update('status_after_filtering', status_after_filtering)
    logger.update('negative_after_post_processing', negative_after_post_processing)

    final_status = check_prediction(gt, pred)
    exec(f'{final_status} = 1')

    log = {}


    log = logger.get_logs(log,'name')
    log = logger.get_logs(log,'bleed')
    log = logger.get_logs(log,'ich')
    log = logger.get_logs(log,'ivh')
    log = logger.get_logs(log,'sah')



    log = logger.get_logs(log, 'status_after_ensemble')
    log = logger.get_logs(log, 'pos_pred_pixels_ensemble')
    log = logger.get_logs(log, 'negative_after_ensemble')
    log = logger.get_logs(log, 'over_lap')
    log = logger.get_logs(log, 'TTA_used')
    log = logger.get_logs(log, 'negative_TTA')
    log = logger.get_logs(log, 'removed_pixels_TTA')
    log = logger.get_logs(log, 'negative_filtering')
    log = logger.get_logs(log, 'pixel_removed_filtering')
    log = logger.get_logs(log, 'status_after_filtering')
    log = logger.get_logs(log, 'negative_after_post_processing')

    log['tp'] = tp
    log['fp'] = fp
    log['tn'] = tn
    log['fn'] = fn





    tp_tot += tp
    fn_tot += fn
    tn_tot += tn
    fp_tot += fp

    
    
    run.log('True Positive',tp_tot)
    run.log('False Positive',fp_tot)
    run.log('True Negative',tn_tot)
    run.log('False Negative',fn_tot)

    smooth = 1e-8


    run.log('Sensitivity',(tp_tot/(tp_tot + fn_tot + smooth)*100))
    
    run.log('Specifisity',(tn_tot / (tn_tot + fp_tot + smooth) * 100))
    
    run.log('Precision',(tp_tot / (tp_tot + fp_tot + smooth) * 100))

    run.log('Negative Predictive Value',(tn_tot / (tn_tot + fn_tot + smooth) * 100))
    run.log('Positive Predictive Value',(tp_tot / (fp_tot + tp_tot + smooth) * 100))

    run.log('False Positive Rate',(fp_tot / (fp_tot + tn_tot + smooth) * 100))
    run.log('False Negative Rate',(fn_tot / (fn_tot + tp_tot + smooth) * 100))
    run.log('False Discovery Rate',(fp_tot / (fp_tot + tp_tot + smooth) * 100))


    
    copy_img = np.clip(image, 0,200)
    copy_img = copy_img.astype('uint8')
    
    copy_pred = pred[...,0].copy()
    copy_pred = copy_pred.astype('uint8')
    
    copy_pred_ensemble = pred_ensemble[...,0].copy()
    copy_pred_ensemble = copy_pred_ensemble.astype('uint8')           


    # Rotate image and segmentations to correct rotation to display the results
    k = -1 
    copy_img           = np.rot90(copy_img, axes = [1,2], k = k)
    copy_pred          = np.rot90(copy_pred, axes = [1,2], k = k)
    copy_pred_ensemble = np.rot90(copy_pred_ensemble, axes = [1,2], k= k)

    img_nifti = nib.Nifti1Image(copy_img,affine)
    seg_nifti = nib.Nifti1Image(copy_pred,affine)
    seg_nifti_ensemble = nib.Nifti1Image(copy_pred_ensemble,affine)


    nib.save(img_nifti,join(f'{folder_name}/{final_status}/{name}.nii.gz'))
    nib.save(seg_nifti,join(f'{folder_name}/{final_status}/{name}_seg.nii.gz'))
    nib.save(seg_nifti_ensemble,join(f'{folder_name}/{final_status}/{name}_seg_ensemble.nii.gz'))

    if pred_TTA is not None: 
        copy_pred_TTA = pred_TTA[...,0].copy().astype('uint8')
        copy_pred_TTA = np.rot90(copy_pred_TTA, axes = [1,2], k = k)
        seg_nifti_TTA = nib.Nifti1Image(copy_pred_TTA,affine)
        nib.save(seg_nifti_TTA,join(f'{folder_name}/{final_status}/{name}_seg_TTA.nii.gz'))



    for key, value in base_preds.items():
        value = np.rot90(value, axes = [1,2], k = -1)
        value = (value > thresh).astype('uint8')
        base_seg = nib.Nifti1Image(value[...,0],affine)
        nib.save(base_seg,join(f'{folder_name}/{final_status}/{name}_{key}.nii.gz'))

    logger.update('duration',time.time() - start_time)
    log = logger.get_logs(log, 'duration')

    file_exists = os.path.isfile(f'{folder_name}/validation_run_log.csv')


    with open(f'{folder_name}/validation_run_log.csv', 'a+') as file:
        writer = csv.DictWriter(file, fieldnames=log.keys())

        if not file_exists:
            writer.writeheader()
        writer.writerow(log)


    
