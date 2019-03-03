# -*- coding: utf-8 -*-

import os
import os.path as osp
import numpy as np
import pprint
from datetime import datetime
import tensorflow as tf
from ops import *
import scipy.io as scio
from glob import glob
import tensorlayer as tl
import time
import matplotlib.pyplot as plt
from skimage import io

path = "SegPre_MRBrain_enlarge_BN"

flags = tf.flags
flags.DEFINE_string("tfrecord_filename", "./data/tfrecords/mrbrains_enlarge_train", "The name of train dataset []")
flags.DEFINE_string("tfrecord_filename_val", "./data/tfrecords/mrbrains_enlarge_val", "The name of val dataset []")  #ckptnum
flags.DEFINE_integer('BATCH_SIZE', 16, 'The size of batch images [128]')
flags.DEFINE_integer('ckptnum', 42000, 'The index of checkpoint for test[128]')
flags.DEFINE_integer('iterations', 42000, 'The number of iteration [128]')
flags.DEFINE_float('LR', 0.001, 'Learning rate of for Optimizer [3e-3]')
flags.DEFINE_integer('NUM_GPUS', 0, 'The number of GPU to use [1]')
flags.DEFINE_integer("size", 240, "width of image  . [1]")
flags.DEFINE_integer("crop_size", 128, "width of crop image  . [1]")
flags.DEFINE_integer("CLASS", 4, "width of image  . [1]")
flags.DEFINE_integer("fil_num", 32, "basic number of convolutional kernel  . [1]")
#flags.DEFINE_integer("repeat_B", 1, "number of block  . [1]")
flags.DEFINE_boolean('IS_TRAIN', True, 'True for train, else test. [True]')
flags.DEFINE_boolean('BN', True, 'True for BatchNormalization. [True]')
flags.DEFINE_boolean('LOAD_MODEL', True,'True for load checkpoint and continue training. [True]')
flags.DEFINE_string('MODEL_DIR', "%s"%(path),
                    'If LOAD_MODEL, provide the MODEL_DIR. [./model/BEGAN/]')

FLAGS = flags.FLAGS


GPU_ID = FLAGS.NUM_GPUS#get_gpu_id(gpu_num = FLAGS.NUM_GPUS)
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.per_process_gpu_memory_fraction = 1
config.gpu_options.allow_growth = False
# network model
def Seg(images, batch_size = FLAGS.BATCH_SIZE, 
        is_train = True, reuse = False, 
        name = 'Segmentation', BN = False,size = FLAGS.size):

    with tf.variable_scope("Net_%s"%(name), reuse = reuse):
        BN = True
        f_num = FLAGS.fil_num
        repeat_ =2
        h1 = images

          
        for i in range(repeat_+1):
            name_ = "Conv_ReLu_%s_1_seg_encoder"%(i) 
            h1 =   tf.nn.relu(Conv2d(h1, output_dim = f_num, name = name_, is_train = is_train, BN = BN))
            h_toshow = h1

        h2_level = h1
        h2 = MaxPooling(h1)
        
        for i in range(repeat_):
            name_ = "Conv_ReLu_%s_2_seg_encoder"%(i) 
            h2 =   tf.nn.relu(Conv2d(h2, output_dim = f_num*2, name = name_, is_train = is_train, BN = BN))
            
        h3_level = h2
        h3 = MaxPooling(h2)

        for i in range(repeat_):
            name_ = "Conv_ReLu_%s_3_seg_encoder"%(i)
            h3 =   tf.nn.relu(Conv2d(h3, output_dim = f_num*2*2, name = name_, is_train = is_train, BN = BN))

        name_ = "DeConv_ReLu_3_seg_decoder"
        h3_ = Deconv2d(h3, output_shape = [batch_size, int(size/2), int(size/2), f_num*2], name = name_)#120

        h3_ = tf.concat([h3_level,h3_], 3)

        for i in range(repeat_):
            name_ = "Conv_ReLu_%s_-2_seg_decoder"%(i)#30
            h3_ =   tf.nn.relu(Conv2d(h3_, output_dim = f_num*2, name = name_, is_train = is_train, BN = BN))           
            
        name_ = "DeConv_ReLu_4_seg_decoder"
        h2_ = Deconv2d(h3_, output_shape = [batch_size, size, size, f_num], name = name_)#120
        h2_ = tf.concat([h2_level,h2_], 3)
        for i in range(repeat_):
            name_ = "Conv_ReLu_%s_-1_seg_decoder"%(i)#30
            h2_ =   tf.nn.relu(Conv2d(h2_, output_dim = f_num, name = name_, is_train = is_train, BN = BN))             
            
        h = Conv2d(h2_, output_dim = FLAGS.CLASS, name = 'Conv2d_-1_seg_decoder', k_h = 1, k_w = 1, is_train = is_train)
        # h=240*240*4
        variables = tf.contrib.framework.get_variables("Net_%s"%(name))
        return h, variables, h_toshow
    

def train(BN = FLAGS.BN, is_train = True):
    
    sess = tf.Session(config = config)
    
    global_step = bias('global_step', [], trainable = False,bias_start = 0)

    im_fs, im_label = read_and_decode()
    im_fs_, im_label_ = read_and_decode_for_val()

    ####

    logit, var_list, _ = Seg(im_fs, is_train = is_train, size = FLAGS.crop_size, BN = BN)

    loss_seg = losses_seg(logit, im_label,FLAGS.CLASS)
    loss = loss_seg  
    
    logit_, _, _ = Seg(im_fs_, is_train = is_train,reuse =True, size = FLAGS.crop_size, BN = BN)

    val_loss_seg = losses_seg(logit_, im_label_,FLAGS.CLASS)
    val_loss = val_loss_seg  
    
    lr = tf.train.exponential_decay(
        learning_rate = FLAGS.LR,
        global_step = global_step,
        staircase = True,
        decay_steps = 10000,
        decay_rate = 0.1
         
    )
#    var_list = tf.trainable_variables()
    

    print( "total vars is : %s"%(cal_para(var_list)))
        
    optim = tf.train.AdamOptimizer(lr).minimize(
        loss,
        var_list = var_list,
        global_step = global_step
    )

    summary_op = tf.summary.merge([
        tf.summary.scalar('Trainloss', loss),
        tf.summary.scalar('loss_seg', loss_seg),
        tf.summary.scalar('lr', lr),
        tf.summary.scalar('Val_loss', val_loss),
        tf.summary.scalar('val_loss_seg', val_loss_seg)
    ])

    init = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    saver = tf.train.Saver(var_list, max_to_keep = None)
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    start = 0
    model_dir = osp.join('models', FLAGS.MODEL_DIR)
    if FLAGS.LOAD_MODEL:
        print(' [*] Reading checkpoints...')
        
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:            
            model_name = "baseline_model.ckpt-%s"%(FLAGS.ckptnum)   
            ckpt_name = os.path.join(model_dir, model_name)   
            saver.restore(sess,  ckpt_name)
            global_step = ckpt_name.split(
                '/')[-1].split('-')[-1]
            print('Loading success, global_step is %s' % global_step)
            start = int(global_step)
        else:
            print(' [*] Failed to find a checkpoint')
            start = 0

    print( '******* start with %d *******' % start)

    idx = start
    start_time = time.time()
    try:
        while not coord.should_stop():
            _, summary_str, lr_val, tain_loss_,loss_seg_,val_loss_,val_loss_seg_ = sess.run([
                optim, summary_op, lr, loss,loss_seg,val_loss,val_loss_seg
            ])
            idx += 1
            writer.add_summary(summary_str, idx)
#            t1_,test_logits_t1_,t2_,test_logits_t2_ = sess.run( [im_t1,train_t1,im_t2,train_t2])
#            print np.mean(t1_), np.mean(test_logits_t1_), np.mean(t2_), np.mean(test_logits_t2_)
            

            if ((idx % 100 == 0) or (idx ==1)):
                

                print(
                    '%s : %d, lr : %.10f \n'
                    'train_loss: %.6f, loss_seg: %.6f \n'
                    'val_loss: %.6f,val_loss_seg: %.6f time: %.4f \n'
                    % (path,idx,lr_val,tain_loss_,loss_seg_, val_loss_,val_loss_seg_,(time.time() - start_time)))

                logit_1 = OneHot_Re(logit)
                logit_1_np, label_np = sess.run([logit_1, im_label])
                save_images_(abs(label_np) , [4, 4], 
                            './{}/_im_label_train_{:04d}.png'.format(sample_dir, idx),mul=False)   
                save_images_(abs(logit_1_np) ,[4, 4], 
                            './{}/_im_logit_train_{:04d}.png'.format(sample_dir, idx),mul=False)   
    
                logit_1 = OneHot_Re(logit_)
                logit_1_np, label_np = sess.run([logit_1, im_label_])
                save_images_(abs(label_np) , [4, 4], 
                            './{}/_im_label_test_{:04d}.png'.format(sample_dir, idx),mul=False)   
                save_images_(abs(logit_1_np) ,[4, 4], 
                            './{}/_im_logit_test_{:04d}.png'.format(sample_dir, idx),mul=False)

                      

            if idx % 1000 == 0:

                checkpoint_path = osp.join(model_dir, 'baseline_model.ckpt')
                saver.save(sess, checkpoint_path, global_step = idx)
                print('**********  baseline_model%d saved  **********' % idx)
            if idx == FLAGS.iterations +1:
                coord.request_stop()                
    except tf.errors.OutOfRangeError:
        checkpoint_path = osp.join(model_dir, 'baseline_model.ckpt')
        saver.save(sess, checkpoint_path, global_step = idx)
        print('**********  baseline_model%d saved  **********' % idx)
        print('Training do1e!')

    finally:
        coord.request_stop()
    print( "stop")

    coord.join(threads)
#    sess.close()

def evaluate(batch_size = 1, size = FLAGS.size, c_dim = 1, is_train = False, BN =  FLAGS.BN):

   
    model_dir = osp.join('models', FLAGS.MODEL_DIR)
    model_name = "baseline_model.ckpt-%s"%(FLAGS.ckptnum)   
    checkpoint_dir = os.path.join(model_dir, model_name)   

    im_ori = tf.placeholder(tf.float32, [batch_size, size, size, 3], name='im_ori')

    logit, var_list, h_toshow  = Seg(im_ori, is_train = is_train, 
                                     size = FLAGS.size, 
                                     batch_size = batch_size)
    logit_1 = OneHot_Re(logit)
    
    print ("build model OK...") 
    sess = tf.Session(config = config)
    saver = tf.train.Saver(var_list)
    saver.restore(sess, checkpoint_dir)
    print ("load model OK...\n")
    print("start generating...")
    paths_in_train_fs = glob('./data/MRbrain_mat_testing/*fs.mat')
    print( paths_in_train_fs)

    for i in range(len(paths_in_train_fs)):
        print( i)
        
        path_fs = paths_in_train_fs[i]
        path_label = "%slabel.mat"%(path_fs[0:-6])
        
        name_fs = path_fs[-11:-4]
        name_label = path_label[-14:-4]
        print( path_fs, name_fs)
        print( path_label,name_label)
        img_fs = scio.loadmat(path_fs)['fs_all'].astype(np.float32)
        label = scio.loadmat(path_label)['label_'].astype(np.uint8)
        label_ =  np.reshape(label,[1,size,size,1])
#        asas

        logit_1_ = sess.run(logit_1, feed_dict = {im_ori: np.reshape(img_fs,[1,size,size,3])} )
        
        scio.savemat('./test/%s/%s_logit.mat'%(path,name_fs),{'logit':np.squeeze(logit_1_)})
        scio.savemat('./test/%s/%s_label.mat'%(path,name_fs),{'label':np.squeeze(label)})
        label_color = draw_seg_map(label_)
        scio.savemat('./test/%s/%s_label_color.mat'%(path,name_fs),{'label_color':label_color})
        logit_color = draw_seg_map(np.squeeze(logit_1_))
        scio.savemat('./test/%s/%s_logit_color.mat' % (path, name_fs), {'logit_color': logit_color})
        
def draw_seg_map(ind):
    ind = np.squeeze(ind)
    R = ind.copy()
    G = ind.copy()
    B = ind.copy()
    w =  [105,139,34]# [218,165,32]
    g =  [218,165,32]  #[105,139,34] 
    c = [139,35,35] # [139,35,35]
    b =[0,0,0]
    label_colors = np.array([b,w,g,c])
    for l in range(0,4):
        R[ind == l] = label_colors[l,0]
        G[ind == l] = label_colors[l,1]
        B[ind == l] = label_colors[l,2]
    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = R/255.0
    rgb[:,:,1] = G/255.0
    rgb[:,:,2] = B/255.0
    return rgb


def getds(x, size=240):
    global mask_complex_1
    global mask_complex_2
    global mask_complex_3
    #    global mask_complex_4
    x = tf.cast(x, tf.complex64)
    x1, x2, x3 = tf.split(x, 3, 3)
    x1 = tf.squeeze(x1)
    x2 = tf.squeeze(x2)
    x3 = tf.squeeze(x3)
    #    x4 = tf.squeeze(x4)
    x1_fft = tf.fft2d(x1) / size
    kd_x1 = x1_fft * mask_complex_1

    x2_fft = tf.fft2d(x2) / size
    kd_x2 = x2_fft * mask_complex_2

    x3_fft = tf.fft2d(x3) / size
    kd_x3 = x3_fft * mask_complex_3

    #    x4_fft = tf.fft2d(x4)/size
    #    kd_x4 = x4_fft*mask_complex_4

    x1_ds = tf.cast(tf.abs(tf.ifft2d(kd_x1) * size), tf.float32)
    x2_ds = tf.cast(tf.abs(tf.ifft2d(kd_x2) * size), tf.float32)
    x3_ds = tf.cast(tf.abs(tf.ifft2d(kd_x3) * size), tf.float32)
    #    x4_ds = tf.cast(tf.abs(tf.ifft2d(kd_x4)*size), tf.float32)
    x1 = tf.expand_dims(x1, -1)
    x2 = tf.expand_dims(x2, -1)
    x3 = tf.expand_dims(x3, -1)
    #    x4 = tf.expand_dims(x4,-1)

    kd_x1 = tf.expand_dims(kd_x1, -1)
    kd_x2 = tf.expand_dims(kd_x2, -1)
    kd_x3 = tf.expand_dims(kd_x3, -1)
    #    kd_x4 = tf.expand_dims(kd_x4,-1)

    x1_ds = tf.expand_dims(x1_ds, -1)
    x2_ds = tf.expand_dims(x2_ds, -1)
    x3_ds = tf.expand_dims(x3_ds, -1)
    #    x4_ds = tf.expand_dims(x4_ds,-1)

    x1 = tf.cast(x1, tf.float32)
    x2 = tf.cast(x2, tf.float32)
    x3 = tf.cast(x3, tf.float32)
    #    x4 = tf.cast(x4, tf.float32)


    return x1, x2, x3, kd_x1, kd_x2, kd_x3, x1_ds, x2_ds, x3_ds


# im_t1,im_t1ce,im_t2,im_flair,kd_t1,kd_t1ce,kd_t2,kd_flair,im_t1_ds,im_t1ce_ds,im_t2_ds,im_flair_ds

def downsample(x, mask_np, size):
    x = x.astype(np.complex64)
    img_1_fft = np.fft.fft2(x) / size
    k_down = img_1_fft * mask_np
    real_ = np.real(k_down).astype(np.float32)
    imag_ = np.imag(k_down).astype(np.float32)
    img_1_ds = abs(np.fft.ifft2(k_down) * size).astype(np.float32)
    return img_1_ds, real_, imag_

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
    
def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))


def read_and_decode():

    filename_queue = tf.train.string_input_producer([FLAGS.tfrecord_filename], shuffle = True)  ######num_epochs need to modify 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
  
    features = tf.parse_single_example(
        serialized_example,
        features={
            'img': tf.FixedLenFeature([], tf.string),
           'label': tf.FixedLenFeature([], tf.string)
        }
    )
    im_fs = tf.reshape(tf.decode_raw(features['img'], tf.float32),[ FLAGS.size, FLAGS.size, 3])
    im_label = tf.reshape(tf.decode_raw(features['label'], tf.uint8),[ FLAGS.size, FLAGS.size, 1])
    
    _time = time.time()
    im_fs = tf.random_crop(im_fs, [FLAGS.crop_size, FLAGS.crop_size, 3], seed = _time)
    im_label = tf.random_crop(im_label, [FLAGS.crop_size, FLAGS.crop_size, 1], seed = _time)

    im_fs, im_label = tf.train.shuffle_batch([im_fs, im_label], 
                                             batch_size = FLAGS.BATCH_SIZE, 
                                             capacity = 300, min_after_dequeue = 200)

    return im_fs, im_label 
def read_and_decode_for_val(): # read tfrecord


    filename_queue = tf.train.string_input_producer([FLAGS.tfrecord_filename_val], shuffle = False)  ######num_epochs need to modify 
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'img': tf.FixedLenFeature([], tf.string),
           'label': tf.FixedLenFeature([], tf.string)
        }
    )
    im_fs = tf.reshape(tf.decode_raw(features['img'], tf.float32), [ FLAGS.size, FLAGS.size, 3])
    im_label = tf.reshape(tf.decode_raw(features['label'], tf.uint8), [ FLAGS.size, FLAGS.size, 1])
    
    _time = time.time()
    im_fs = tf.random_crop(im_fs, [FLAGS.crop_size, FLAGS.crop_size, 3], seed = _time)
    im_label = tf.random_crop(im_label, [FLAGS.crop_size, FLAGS.crop_size, 1], seed = _time)
    
    im_fs, im_label = tf.train.batch([im_fs, im_label], batch_size = FLAGS.BATCH_SIZE)
    return im_fs, im_label 


if __name__ == '__main__':
    log_dir = osp.join('logs', FLAGS.MODEL_DIR)
    model_dir = osp.join('models', FLAGS.MODEL_DIR)
    test_dir = osp.join('test', FLAGS.MODEL_DIR)
    sample_dir = osp.join('samples', FLAGS.MODEL_DIR)


    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)
    if not osp.exists(sample_dir):
        os.makedirs(sample_dir)
    if not osp.exists(test_dir):
        os.makedirs(test_dir)
    print('Current time: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('The network initialization with learning rate %f ...' % FLAGS.LR)
    pprint.pprint(FLAGS.__flags)
    if FLAGS.IS_TRAIN == True:
        train()
    else:
        evaluate()
    