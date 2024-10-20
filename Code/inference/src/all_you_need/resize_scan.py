import numpy as np
import cv2

def resize_scan(image):

    image = image.copy()

    empty = []
    if image[0].shape == (512,512,1):
        pass
    
    elif image[0].shape == (512,512):
        image = np.expand_dims(image,-1)

    else:
        
        for b in range(len(image)):
            x = image[b,:,:]
            x = cv2.resize(x,(512,512))
            x = np.expand_dims(x,2)
            empty.append(np.expand_dims(x,0))
        image = np.concatenate(empty)
    
    return image