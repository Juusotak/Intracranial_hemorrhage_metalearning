import tensorflow as tf
import cv2
import numpy as np
from scipy import stats, ndimage
import os
from   os import listdir
from   os.path import join
import pydicom
from typing import Union



def create_multilabel_segmentation(image_list):
    result = np.zeros_like(image_list[0])
    for index, img in enumerate(image_list):
        result = np.where(img > 0.5, index + 1, result)
    return result

    



def check_prediction(gt, prediction):
    
    if gt.lower() == 'neg': 
        bleed = False
    else:
        bleed = True
    
    if np.count_nonzero(prediction) == 0 and not bleed:
        return 'tn'
    elif np.count_nonzero(prediction) >= 1 and not bleed:
        return 'fp'
    elif np.count_nonzero(prediction) == 0 and  bleed:
        return 'fn'
    else:
        return 'tp'


def filter_plane(ds, plane = 'axial'):

    if plane.lower() == 'axial': plane = 2
    orientation = ds.ImageOrientationPatient
    cross = abs(np.cross(orientation[0:3], orientation[3:6]))
    
    if np.argmax(cross) == plane:
        return ds
    else:
        return None
    

    


def read_dicom(series_path):
    dicom_files = []

    files = os.listdir(series_path)
    files = [join(series_path, file) for file in files]
    # slices = [pydicom.dcmread(file) for file in files]
    slices = []

    for file in files:
        try:
            slices.append(pydicom.dcmread(file))
        except pydicom.errors.InvalidDicomError:
            pass    
    
    try:
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key = lambda x: float(x.InstanceNumber))
    
    
    pixel_array = [(x.pixel_array*x.RescaleSlope + x.RescaleIntercept) for x in slices]
    

    output = False  
    try:
        pixel_array = np.stack(pixel_array, axis = 0)
        output = True
    except ValueError:
        pass
    
    if not output:
        try:    
            pixel_array = np.stack(pixel_array[:-1], axis = 0)
            output = True
        except ValueError:
            pass
        
    if not output:   
        try:    
            pixel_array = np.stack(pixel_array[1:], axis = 0)
        except ValueError:
            pass

    return pixel_array


def read_dicom_2(series_path, strict_axial = False, slice_thickness: Union[float,None] = 1.0):

    """ Function for loading the dicom series from folder. 

    Arguments: 
        series_path = path to the folder containing the dicom series
        strict_axisl = False or True. If False: the functions searches the imaging plane with most slices in the series and selects the images wiht the plan. True: the function only accepts axial slices.
        slice_thickness = Float or None, if float the functions selects only images with slice_thickness greater than the value. If None filtering is not applied. 


    """

    dicom_files = []

    files = os.listdir(series_path)
    files = [join(series_path, file) for file in files]
    # slices = [pydicom.dcmread(file) for file in files]
    slices = []

    for file in files:
        try:
            slices.append(pydicom.dcmread(file))
        except pydicom.errors.InvalidDicomError:
            pass 

    if strict_axial:
        slices = [filter_plane(ds, plane = 'axial') for ds in slices]
        slices = [x for x in slices if x is not None]
    
    else:
        orientations = [ds.ImageOrientationPatient for ds in slices]
        orientations = [abs(np.cross(orientation[:3],orientation[3:])) for orientation in orientations]
        orientations = [np.argmax(cross) for cross in orientations]
        
        values, counts = np.unique(orientations, return_counts = True)
        plane = values[counts.argmax()]
        slices = [sl for sl,pl in zip(slices,orientations) if pl == plane]

    if slice_thickness is not None:

        slices = [ds for ds in slices if ds.SliceThickness > 1.0]

    
    
    try:
        slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    except AttributeError:
        slices.sort(key = lambda x: float(x.InstanceNumber))
    
    
    pixel_array = [(x.pixel_array*x.RescaleSlope + x.RescaleIntercept) for x in slices]
    

    output = False  
    try:
        pixel_array = np.stack(pixel_array, axis = 0)
        output = True
    except ValueError:
        pass
    
    if not output:
        try:    
            pixel_array = np.stack(pixel_array[:-1], axis = 0)
            output = True
        except ValueError:
            pass
        
    if not output:   
        try:    
            pixel_array = np.stack(pixel_array[1:], axis = 0)
        except ValueError:
            pass

    return pixel_array