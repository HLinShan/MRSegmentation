# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import scipy.misc
import tensorflow.contrib.layers as ly
def int_shape(tensor):
    shape = tensor.get_shape().as_list()
    return [num if num is not None else -1 for num in shape]
def get_conv_shape(tensor):
    shape = int_shape(tensor)
    return shape
def up_scale(
        x, bilinear = False, scale = 2
    ):
    _,h,w,_ = get_conv_shape(x)
    new_h = h*scale
    new_w = w*scale
    new_h = tf.cast(new_h, tf.int32)
    new_w = tf.cast(new_w, tf.int32)
    if bilinear:
        x = tf.image.resize_bilinear(x, (new_h, new_w))
    else:
        x = tf.image.resize_nearest_neighbor(x, (new_h, new_w))
    return x

def losses_seg(logits, labels, num_classes=2, head=None):
    tf.losses.sparse_softmax_cross_entropy


    with tf.name_scope('losses_seg'):
#        print logits, labels
        
        logits = tf.cast(tf.reshape(logits, (-1, num_classes)), tf.float32)

        labels = tf.cast(tf.reshape(labels, [-1]), tf.int64)
#        print logits, labels
        
        softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = labels, logits = logits)
#        tf.nn.weighted_cross_entropy_with_logits
        loss = tf.reduce_mean(softmax, name='entropy_mean')

    return loss
def bound(x,min_= 0., max_ = 1.):
    x[np.where(x < min_ )] = min_
    x[np.where(x > max_ )] = max_
    return x
def cal_para(var_list):
    total_var = 0
    for var in var_list:
        shape = var.get_shape()
        var_p = 1
        for dim in shape:
            var_p *= dim.value
        total_var += var_p
    return total_var
def bias(name, shape, bias_start = 0.1, trainable = True):
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
        initializer = tf.constant_initializer(bias_start, dtype = tf.float32))
    return var

def compute_psnr(im1, im2):
    diff = np.abs(im1- im2)
    diff = diff.astype(np.float32)
    rmse = np.square(diff).mean()
    rmse = np.sqrt(rmse)
    psnr = 20*np.log10(1/rmse)
    return psnr     
def weight(name, shape, stddev = 0.02, trainable = True):
    var = tf.get_variable(name, shape, tf.float32, trainable = trainable,
        initializer = tf.contrib.layers.xavier_initializer(dtype = tf.float32), 
         regularizer=tf.contrib.layers.l2_regularizer(scale=1e-4))
    return var
def data_fid(input_, k_down, mask_complex,batch_size,size_ = 256.):
    input_shape = input_.get_shape()
    k_down = tf.squeeze(k_down)
    input_ = tf.squeeze(input_)

    result_ = (1e6 * k_down + (tf.fft2d(tf.cast(input_,tf.complex64))/size_)) /( (1e6 * mask_complex )+1)
    result_ = tf.abs(tf.ifft2d(result_)*size_)
    
    result_ = tf.reshape(result_, input_shape)
    return result_

def OneHot(value, depth = 8):
    return tf.one_hot(value, depth, dtype = tf.float32)

def OneHot_Re(value):
    #  axis max of the axis=3 the
    x = tf.argmax(value, axis =3)
    # expand add the dim of 3chanel
    result = tf.expand_dims(x, 3)
    return result

def InnerProduct(value, output_shape, name = 'InnerProduct', with_w = False):
    with tf.variable_scope(name):
        shape = value.get_shape().as_list()
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)

        if with_w:
            return tf.matmul(value, weights) + biases, weights, biases
        else:
            return tf.matmul(value, weights) + biases


def LReLU(x, leak = 0.2, name = 'LReLU'):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name = name)


def SeLU(value, name = 'SeLU'):
    with tf.variable_scope(name):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(value >= 0.0, value, alpha * tf.nn.elu(value))


def ELU(value, name = 'ELU'):
    with tf.variable_scope(name):
        return tf.nn.elu(value)


def ReLU(value, name = 'ReLU'):
    with tf.variable_scope(name):
        return tf.nn.relu(value)


def Deconv2d(value, output_shape, k_h = 3, k_w = 3, strides =[1, 2, 2, 1],
             name = 'Deconv2d', with_w = False):
    with tf.variable_scope(name):
        weights = weight(name+'weights',
                         [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights,
                                        output_shape, strides = strides)
        biases = bias(name+'biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv

def Conv2d_dil(value, output_dim,rate=1, k_h = 3, k_w = 3,
           strides =[1, 1, 1, 1], name = 'Conv2d',is_train = True, BN= False):
    
    
#    conv = ly.conv2d(value, num_outputs =output_dim, kernel_size = [k_h,h_w],stride = strides,activation_fn=None,normalizer_fn=None )
    with tf.variable_scope(name):
        weights = weight(name+'weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.atrous_conv2d(value, weights, rate = rate, padding = 'SAME')
#        conv = tf.nn.conv2d(value, weights, strides = strides, padding = 'SAME')
        biases = bias(name+'biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        if BN:
            conv =BatchNorm(conv, is_train = is_train)
        else:
            conv = conv

        return conv

    
def Conv2d(value, output_dim, k_h = 3, k_w = 3,
           strides =[1, 1, 1, 1], name = 'Conv2d',is_train = True, BN= False):
    
    
#    conv = ly.conv2d(value, num_outputs =output_dim, kernel_size = [k_h,h_w],stride = strides,activation_fn=None,normalizer_fn=None )
    with tf.variable_scope(name):
        weights = weight(name+'weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides = strides, padding = 'SAME')
        biases = bias(name+'biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        
        if BN:
            conv =BatchNorm(conv, is_train = is_train)
        else:
            conv = conv

        return conv

def addcoords(inputs, x_dim=64, y_dim=64, with_r = False):
    batch_size = tf.shape(inputs)[0]
    xx_ones = tf.ones([batch_size, x_dim], dtype=tf.int32)
    xx_ones = tf.expand_dims(xx_ones, -1)
    xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0),
                       [batch_size, 1])
    xx_range = tf.expand_dims(xx_range, 1)

    xx_channel = tf.matmul(xx_ones, xx_range)
    xx_channel = tf.expand_dims(xx_channel, -1)

    yy_ones = tf.ones([batch_size, y_dim], dtype=tf.int32)
    yy_ones = tf.expand_dims(yy_ones, 1)
    yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0),
                       [batch_size, 1])
    yy_range = tf.expand_dims(yy_range, -1)

    yy_channel = tf.matmul(yy_range, yy_ones)
    yy_channel = tf.expand_dims(yy_channel, -1)

    xx_channel = tf.cast(xx_channel, 'float32') / (x_dim - 1)
    yy_channel = tf.cast(yy_channel, 'float32') / (y_dim - 1)
    xx_channel = xx_channel*2 -1
    yy_channel = yy_channel*2 -1

    ret = tf.concat([inputs, xx_channel, yy_channel], axis=-1)

    if with_r:
        rr = tf.sqrt(tf.square(xx_channel - 0.5) + tf.square(yy_channel - 0.5))
        ret = tf.concat([ret, rr], axis=-1)

    return ret

def CoordConv(
        inputs, output_dim, k_h = 3, k_w = 3,
        strides =[1, 1, 1, 1], name = 'CoordConv', is_train = True, BN=False
    ):
    with tf.variable_scope(name):
        _, height, width, _ = map(lambda i: i.value, inputs.get_shape())
        ret = addcoords(inputs, height, width)
        name_ = 'Conv'
        ret = Conv2d(ret, output_dim, k_h, k_w, strides, name = name+name_, is_train=is_train, BN=BN)
        return ret

def Conv2d_ly(value, output_dim, k_h = 3, k_w = 3,
           strides =[1, 1, 1, 1], name = 'Conv2d',is_train = True, BN= False,wd = 1e-4):
    
    norm_params={
            'is_training' : is_train,
            'updates_collections':None,
            'scale':True
            }
    if BN:
      conv = ly.conv2d(value, num_outputs =output_dim, kernel_size = [k_h,k_w],stride = 1,
                     weights_regularizer =ly.l2_regularizer(wd),activation_fn=None,normalizer_fn=ly.batch_norm,normalizer_params = norm_params, scope = name )
    else:
      conv = ly.conv2d(value, num_outputs =output_dim, kernel_size = [k_h,k_w],stride = 1,
             weights_regularizer =ly.l2_regularizer(wd),activation_fn=None,normalizer_fn=None,scope = name )
 
    return conv

def MaxPooling(value, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1],
               padding = 'SAME', name = 'MaxPooling'):
    with tf.variable_scope(name):
        return tf.nn.max_pool(value, ksize = ksize,
                              strides = strides, padding = padding)


def AvgPooling(value, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
               padding = 'SAME', name = 'AvgPooling'):
    with tf.variable_scope(name):
        return tf.nn.avg_pool(value, ksize = ksize,
                              strides = strides, padding = padding)


def Concat(value, cond, name = 'concat'):
    """
    Concatenate conditioning vector on feature map axis.
    """

    with tf.variable_scope(name):
        value_shapes = value.get_shape().as_list()
        cond_shapes = cond.get_shape().as_list()
        return tf.concat([value,
              cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], axis = 3)


def BatchNorm(value, is_train = True, name = 'BatchNorm',
              epsilon = 1e-5, momentum = 0.9):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(
            value,
            decay = momentum,
            updates_collections = None,
            epsilon = epsilon,
            scale = True,
            is_training = is_train,
            scope = name
        )


def GlobalAvePooling(value, name = 'GlobalAvePooling'):
    with tf.variable_scope(name):
        assert value.get_shape().ndims == 4
        return tf.reduce_mean(value, [1, 2])


def ResizeNearNeighbor(value, scale = 2, name = 'Resize'):
    with tf.variable_scope(name):
        _, h, w, _ = value.get_shape().as_list()
        return tf.image.resize_nearest_neighbor(
            value, [h * scale, w * scale], name = name)

def save_images_(images, size, image_path, bound_ = True,mul=False):
    images = images * 255.0
#    images = (images *127.5)+127.5
    if bound_:
        images[np.where(images < 0 )] = 0.
        images[np.where(images > 255 )] = 255.
    else:
        pass
    images = images.astype(np.uint8)    
    return scipy.misc.imsave(image_path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img