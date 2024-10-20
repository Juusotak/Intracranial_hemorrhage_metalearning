import tensorflow as tf
import nilearn.image
import numpy as np
import cv2
import nibabel as nib

model_dir = 'niftynet/'

class nifty_net(): 
    def __init__(self, model_dir):
 
        self.sess = tf.compat.v1.Session(graph=tf.Graph())
        serve = tf.compat.v1.saved_model.load(self.sess ,tags={'serve'}, export_dir=model_dir) 
        self.input_tensor_name = self.extract_input_name(serve.signature_def, self.sess.graph)
        self.output_tensor_name = self.extract_output_name(serve.signature_def, self.sess.graph)
        
        
    def rescale_pred(self, pred, size = 512):
       
        empty = []
        pred = pred.squeeze(-1)
        pred = pred.astype('float')

        
        
                    
        for b in range(len(pred)):
            x = pred[b,:,:]

            x = cv2.resize(x,(size,size))
            x = np.expand_dims(x,2)
            empty.append(np.expand_dims(x,0))
        pred = np.concatenate(empty)
        
            
        return pred
        
    
    def predict(self, img):
 
        if isinstance(img, nib.nifti1.Nifti1Image):
            x = self.preprocess_nifti(img)
        elif isinstance(img, np.ndarray):
            x = self.preprocess_np_array(img)
        else:
            print('invalid image datatype, image should be Nifti of Numpy array')
            
        result = self.sess.run(self.output_tensor_name, feed_dict={self.input_tensor_name: x})
        # result = rescale(result, scale = (1,2,2,1), mode = 'constant', preserve_range=True,anti_aliasing=True)
        
        result = self.rescale_pred(result, size = 512)
        
        return result
    
    
    def get_original_image(self,img,clip=(0, 150)):

        if isinstance(img, nib.nifti1.Nifti1Image):
            x = self.preprocess_nifti(img, downsample=1, size = 512,clip=(0, 150))
        elif isinstance(img, np.ndarray):
            x = self.preprocess_np_array(img, size = 512,clip=(0, 150))
        else:
            print('invalid image datatype, image should be Nifti of Numpy array')
               
        return x
        
        
    def preprocess_nifti(self,img, downsample=2, size=256, clip=(0, 150), dtype=np.int16):
 
        new_affine = img.affine.copy()
        new_affine[:3, :3] = np.matmul(img.affine[:3, :3], np.diag((downsample, downsample, 1)))
        min_value = img.get_fdata().min()
        tmp_img = nilearn.image.resample_img(img, target_affine=new_affine,
            target_shape=(size, size, img.shape[2]), fill_value=min_value)
        data = tmp_img.get_fdata()
        data = np.transpose(data,[2,0,1])
        data = data[...,np.newaxis]
        if clip:
            data = data.clip(min=clip[0], max=clip[1])
            
        return data # nib.Nifti1Image(data.astype(dtype), tmp_img.affine)
    
    
    def preprocess_np_array(self,img,size=256, clip=(0, 150), dtype=np.int16):
 
        image = img
        empty = []
        
        if image[0].shape == (size,size,1):
            image = image
        else:
            if len(img.shape) == 4:
                image = image.squeeze(-1)
            image = image.astype('float')
            for b in range(len(image)):
                x = image[b,:,:]
                x = cv2.resize(x,(size,size))
                x = np.expand_dims(x,2)
                empty.append(np.expand_dims(x,0))
            image = np.concatenate(empty)
            
        if clip:
            image = image.clip(min=clip[0], max=clip[1])
            
        return image.astype(dtype)

    def extract_tensors(self,signature_def, graph):
        
        output = dict()
        for key in signature_def:
            value = signature_def[key]
            if isinstance(value,  tf.compat.v1.TensorInfo):
                output[key] = graph.get_tensor_by_name(value.name)
                
        return output

    def extract_input_name(self,signature_def, graph):
        
        input_tensors = self.extract_tensors(signature_def['serving_default'].inputs, graph)
        #Assuming one input in model.
        key = list(input_tensors.keys())[0]
        
        return input_tensors.get(key).name

    def extract_output_name(self,signature_def, graph):
        
        output_tensors = self.extract_tensors(signature_def['serving_default'].outputs, graph)
        #Assuming one output in model.
        key = list(output_tensors.keys())[0]
        
        return output_tensors.get(key).name
        
