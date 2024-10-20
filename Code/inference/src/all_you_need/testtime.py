import numpy as np
from scipy import ndimage 



def preprocess_slice(data):
    slice1 = np.expand_dims(data,0)
    slice2 = np.expand_dims(np.flipud(data),0)
    slice3 = np.expand_dims(np.fliplr(data),0)
    slice4 = np.expand_dims(np.flipud(np.fliplr(data)),0)
    return np.concatenate((slice1,slice2,slice3,slice4),axis = 0)

def get_results(data, thresh=0.5):
    slice1 = np.expand_dims(data[0],0)
    slice2 = np.expand_dims(np.flipud(data[1]),0)
    slice3 = np.expand_dims(np.fliplr(data[2]),0)
    slice4 = np.expand_dims(np.flipud(np.fliplr(data[3])),0)
    return ((slice1+slice2+slice3+slice4) / 4) > thresh



def remove(pred,n):
    
    out = np.zeros_like(pred)
    pos_sl = np.where(pred.reshape(pred.shape[0],-1).max(-1) != 0)[0]

    for sl in pos_sl:
        for ch in range(out.shape[-1]):

            labeled, _ = ndimage.label(pred[sl,...,ch], structure = np.ones((3,3)))

            values, counts = np.unique(labeled, return_counts = True)
            values, counts = values[1:], counts[1:]

            for value, count in zip(values, counts):
                if count > n:
                    out[sl,...,ch] += labeled == value

    return out


def TTA_function(image,pred, model):
        
    empty = []
    for a in range(len(image)):
        if np.count_nonzero(pred[a])>0:
            data = preprocess_slice(image[a])
            pred2  = model.predict(data)
            if isinstance(pred2, tuple):
                pred2 = pred2[0]

            if pred2.shape[-1] > 3:
                pred2 = pred2[...,1:] #if softmax model is used, else line not needed.
            pred2 = get_results(pred2)

            pred2 = pred2.max(-1, keepdims = True)
            pred2 = remove(pred2,10)
            empty.append(pred2)
        else:
            empty.append(pred[[a]])

    return np.concatenate(empty, 0)




















