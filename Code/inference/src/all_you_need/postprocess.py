from scipy import ndimage
import numpy as np

def postprocess(pred):
    
    out = np.zeros_like(pred)

    for sl in range(pred.shape[0]):
        for ch in range(pred.shape[-1]):

            if np.count_nonzero(pred[sl,...,ch]) < 10:
                pass
            else:
                out[sl,...,ch] = pred[sl,...,ch]

    return out

def postprocess_single_channel(pred, thresh = 0.5):
    out = np.zeros(pred.shape)
    
    for i in range(len(pred)):
        if np.count_nonzero(pred[i]>thresh) < 10:
            pass
        else:
            out[i] = pred[i]

    return out


def compare_base_ensemble(pred, base_pred_sum):
    pred = pred.max(axis= -1).astype('int')
    base_pred_sum = base_pred_sum.max(axis = -1).astype('int')

    over_lap = False

    for i in range(len(base_pred_sum)):
        array, p = ndimage.label(base_pred_sum[i])
        
        for p_v in range(1, p + 1):
            test_array = np.where(array == p_v,1,0)

            x,y = np.where(test_array != 0)

            x_coord = slice( int(x.min()*0.95), int(x.max()*1.05))
            y_coord = slice( int(y.min()*0.95), int(y.max()*1.05))
            over_lap = np.count_nonzero(pred[i,x_coord,y_coord]) > 10
            
    return over_lap


def filter_basesegmentation_ensemble(final_seg, base_seg):

    final_seg = final_seg.copy().astype('int')
    base_seg = base_seg.copy().astype('int')

    output = np.zeros_like(final_seg)
    
    pos_slices = np.where(final_seg.reshape(final_seg.shape[0], -1).max(-1) != 0)[0]
    
    array_base = np.zeros_like(base_seg)
    array_base[pos_slices] = base_seg[pos_slices]

    test_array = np.concatenate([final_seg*2, base_seg], -1).max(-1, keepdims = True)
    

    for sl in pos_slices:
        for ch in range(output.shape[-1]):

            labeled, _ = ndimage.label(test_array[sl,...,ch] > 0, structure = np.ones((3,3)))

            values, counts = np.unique(labeled, return_counts = True)

            values, counts = values[1:], counts[1:]

            for value in values:
                if 2 in test_array[sl, labeled == value, ch]:
                    output[sl,labeled == value,ch] += array_base[sl, labeled == value,ch].astype('int')

        
    return output.clip(0,1)











