import os 
import tensorflow as tf 
from PIL import Image
#from utils import *
import numpy as np
import scipy.io as scio
from glob import glob
import matplotlib.pyplot as plt
import tensorlayer as tl
import random 
from skimage import io

cwd = os.getcwd()
# mask_mat = scio.loadmat('./1D_20_Cart.mat')
# mask_np = mask_mat['mask']
size = 240

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def create_record():
    writer = tf.python_io.TFRecordWriter("./data/tfrecords/mrbrains_enlarge_train")
    count = 0
    paths_in_train_fs = glob('./data/MRbrain_mat_train/*fs.mat')
    train = True
    if train:
        for i in range(int(len(paths_in_train_fs))):
            print (i)
            path_fs = random.choice(paths_in_train_fs)
            path_label = "%slabel.mat"%(path_fs[0:-6])
            image = scio.loadmat(path_fs)['fs_all'].astype(np.float32)
            label = scio.loadmat(path_label)['label_'].astype(np.uint8)
            writer = writer_write(writer,image,label)
    #        plt.figure('1')
    #        plt.imshow(image)
    #        plt.figure('1.5')
    #        plt.imshow(label)
            
            image1 = image[:,:,0] 
            image2 = image[:,:,1] 
            image3  = image[:,:,2] 
            image1 = np.fliplr(image1)
            image2 = np.fliplr(image2)
            image3 = np.fliplr(image3)
            label_ = np.fliplr(label)
            image_ = np.concatenate([np.expand_dims(image1,-1),np.expand_dims(image2,-1),np.expand_dims(image3,-1)],2)        
            writer = writer_write(writer,image_,label_)
    #        plt.figure('2')
    #        plt.imshow(image_)
    #        plt.figure('3')
    #        plt.imshow(label_)
            
            image1 = image[:,:,0] 
            image2 = image[:,:,1] 
            image3  = image[:,:,2] 
            image1 = np.flipud(image1)
            image2 = np.flipud(image2)
            image3 = np.flipud(image3)
            label_ = np.flipud(label)
            image_ = np.concatenate([np.expand_dims(image1,-1),np.expand_dims(image2,-1),np.expand_dims(image3,-1)],2)        
            writer = writer_write(writer,image_,label_)
    #        plt.figure('2')
    #        plt.imshow(image_)
    #        plt.figure('3')
    #        plt.imshow(label_)
    #        asas
            image1 = np.expand_dims(image[:,:,0],-1)
            image2 = np.expand_dims(image[:,:,1],-1)
            image3  = np.expand_dims(image[:,:,2],-1)
            label_ = np.expand_dims(label,-1)
            [image1,image2,image3,label_] = tl.prepro.rotation_multi([image1,image2,image3,label_], 
                                                                    rg=20, is_random=True, fill_mode='constant')
            image_ = np.concatenate([image1,image2,image3],2)
            label_ = np.squeeze(label_).astype(np.uint8)  
            writer = writer_write(writer,image_,label_)
    #        plt.figure('2')
    #        plt.imshow(image_)
    #        plt.figure('3')
    #        plt.imshow(label_)
    #        asas
            image1 = np.expand_dims(image[:,:,0],-1)
            image2 = np.expand_dims(image[:,:,1],-1)
            image3  = np.expand_dims(image[:,:,2],-1)
            label_ = np.expand_dims(label,-1)
            [image1,image2,image3,label_] = tl.prepro.elastic_transform_multi([image1,image2,image3,label_],
                                                                               alpha=720, sigma=24, is_random=True)
            image_ = np.concatenate([image1,image2,image3],2)
            label_ = np.squeeze(label_).astype(np.uint8)  
            writer = writer_write(writer,image_,label_)
    #        plt.figure('2')
    #        plt.imshow(image_)
    #        plt.figure('3')
    #        plt.imshow(label_)
    #        asas
    else:
        for i in range(1):#int(len(paths_in_train_fs))):
            print (i)
            path_fs = random.choice(paths_in_train_fs)
            path_label = "%slabel.mat"%(path_fs[0:-6])
            image = scio.loadmat(path_fs)['fs_all'].astype(np.float32)
            label = scio.loadmat(path_label)['label_'].astype(np.uint8)
            writer = writer_write(writer,image,label) 
            plt.figure('1')
            plt.imshow(image)
            plt.figure('1.5')
            plt.imshow(label)

    writer.close() 
    print ("total %d "%(count))
#def get_enlarged(image,label):
#
#        return image_, label_
    
# def downsample(x):
#     x = x.astype(np.complex64)
#     img_1_fft = np.fft.fft2(x) /size
#     k_down = img_1_fft *mask_np
#     real_ = np.real(k_down).astype(np.float32)
#     imag_ = np.imag(k_down).astype(np.float32)
#     img_1_ds = abs(np.fft.ifft2(k_down)*size).astype(np.float32)
#     img_1_ds = np.squeeze(img_1_ds)
#     real_ = np.squeeze(real_)
#     imag_ = np.squeeze(imag_)
#     return img_1_ds, real_, imag_

def writer_write(writer,image,clean):


        img = image.tobytes()
        label = clean.tobytes()
#        img_ds = img_ds.tobytes()
#        kd_r = kd_r.tobytes()
#        kd_i = kd_i.tobytes()
        example = tf.train.Example( features = tf.train.Features(
        feature = {

                "img":_bytes_feature(img),
                "label":_bytes_feature(label),
#                "img_ds":_bytes_feature(img_ds),
#                "kd_r":_bytes_feature(kd_r),
#                "kd_i":_bytes_feature(kd_i)

                    }))
        writer.write(example.SerializeToString())

        
        return writer      

    
if __name__ == '__main__':
    create_record()
