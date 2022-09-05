# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 11:37:09 2021

@author: salim
"""

from datetime import datetime
import math
import cv2
import glob
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import vstack, hstack
from numpy.random import randn, rand
from numpy.random import randint, permutation
import os
import tensorflow as tf
from tensorflow.keras import optimizers
# from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import numpy as np 
import os
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from skimage.metrics import structural_similarity as ssim

## convert RGB to the personal LAB (LAB2) 
# the input R,G,B,  must be 1D from 0 to 255 
# the outputs are 1D  L [0 1], a [-1 1] b [-1 1]
def RGB2LAB2(R0, G0, B0):
    import numpy as np
    R=R0/255
    G=G0/255
    B=B0/255
    
    Y=0.299*R + 0.587*G + 0.114*B
    X=0.449*R + 0.353*G + 0.198*B
    Z=0.012*R + 0.089*G + 0.899*B  
    
    # X - Y = 0.150*R - 0.234*G + 0.084*B  = a0
    # Y - Z = 0.287*R + 0.498*G - 0.785*B  = b0
    
    L = Y
    a = (X - Y)/0.234
    b = (Y - Z)/0.785
    
    return L, a, b

## convert the personal LAB (LAB2)to the RGB 
# the input L,a,b,  must be 1D L [0 1], a [-1 1] b [-1 1]
# the outputs are 1D  R g B [0 255]
def LAB22RGB(L, a, b):
    import numpy as np
    a11 = 0.299
    a12 = 0.587
    a13 = 0.114
    a21 = (0.15/0.234)
    a22 = (-0.234/0.234)
    a23 = (0.084/0.234)
    a31 = (0.287/0.785)
    a32 = (0.498/0.785)
    a33 = (-0.785/0.785)
    
    aa=np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])
    C0=np.zeros((L.shape[0],3))
    C0[:,0]=L[:,0]
    C0[:,1]=a[:,0]
    C0[:,2]=b[:,0]
    C = np.transpose(C0)
    
    X = np.linalg.inv(aa).dot(C)
    X1D=np.reshape(X,(X.shape[0]*X.shape[1],1))
    p0=np.where(X1D<0)
    X1D[p0[0]]=0
    p1=np.where(X1D>1)
    X1D[p1[0]]=1
    Xr=np.reshape(X1D,(X.shape[0],X.shape[1]))
    
    Rr = Xr[0][:]
    Gr = Xr[1][:]
    Br = Xr[2][:]
    
    R = np.uint(np.round(Rr*255))
    G = np.uint(np.round(Gr*255))
    B = np.uint(np.round(Br*255))
    return R, G, B

def psnr(img1, img2):
    mse = np.mean( (img1.astype("float") - img2.astype("float")) ** 2 )
    # print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(imageA, imageB, bands):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def mae(imageA, imageB, bands):
 	# the 'Mean Squared Error' between the two images is the
 	# sum of the squared difference between the two images;
 	# NOTE: the two images must have the same dimension
 	err = np.sum(np.abs((imageA.astype("float") - imageB.astype("float"))))
 	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
 	return err
    
def rmse(imageA, imageB, bands):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
	err = np.sqrt(err)
	return err
    
def unet1(input_size):
    inputs = Input(input_size)  # 0
    # layers 1-5
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) # 1
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)  # 2
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # 3
    # layers 4-6
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1) # 4
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) # 5
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # 6
    # layers 7-10
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2) # 7
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) # 8
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) # 9
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # 10
    # layers 11-14
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3) # 11
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # 12
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) # 13
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # 14
    # layers 15-18
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4) # 15
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) # 16
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) # 17
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5) # 18
    
    # layers 19-21
    conv55 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool5) # 19
    conv55 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv55) # 20
    conv55 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv55) # 21

    # layers 22-26
    up66 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv55)) # 22+23
    merge66 = concatenate([conv5,up66], axis = 3) # 24
    conv66 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge66) # 25
    conv66 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv66) # 26
    
    # layers 27-31
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv66))# 27+28
    merge6 = concatenate([conv4,up6], axis = 3) # 29
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6) # 30
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6) # 31

    # layers 32-36
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6)) # 32+33
    merge7 = concatenate([conv3,up7], axis = 3) # 34
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7) # 35
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) # 36

    # layers 37-41
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7)) # 37+38
    merge8 = concatenate([conv2,up8], axis = 3) # 39
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8) # 40
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8) # 41
    
    # layers 42-46
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8)) # 42+43
    merge9 = concatenate([conv1,up9], axis = 3) # 44
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9) # 45
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) # 46

    # Up_f01 = conv1    
    # Up_f02 = UpSampling2D(size = (2,2))(conv2)
    # Up_f03 = UpSampling2D(size = (4,4))(conv3)
    # Up_f04 = UpSampling2D(size = (8,8))(conv4)
    # Up_f05 = UpSampling2D(size = (16,16))(conv5)
    # Up_f06 = UpSampling2D(size = (32,32))(conv55)
    # Up_f16 = UpSampling2D(size = (32,32))(conv66)
    # Up_f15 = UpSampling2D(size = (16,16))(conv6)
    # Up_f14 = UpSampling2D(size = (8,8))(conv7)
    # Up_f13 = UpSampling2D(size = (4,4))(conv8)
    # Up_f12 = UpSampling2D(size = (2,2))(conv9)
    # Up_f11 = conv10
    
    Up_f01 = conv1     # 
    Up_f02 = UpSampling2D(size = (2,2))(conv2) # 47
    # Up_f03 = UpSampling2D(size = (4,4))(conv3) # 49
    # Up_f04 = UpSampling2D(size = (8,8))(conv4) # 40
    # Up_f05 = UpSampling2D(size = (16,16))(conv5) # 50
    # Up_f06 = UpSampling2D(size = (32,32))(conv55) # 51
    # Up_f15 = UpSampling2D(size = (16,16))(conv66) # 52
    # Up_f14 = UpSampling2D(size = (8,8))(conv6) # 53
    # Up_f13 = UpSampling2D(size = (4,4))(conv7) # 54
    Up_f12 = UpSampling2D(size = (2,2))(conv8) # 48
    # Up_f12 = UpSampling2D(size = (2,2))(conv9) 
    Up_f11 = conv9  #  conv10     # 56

    # merge11 = concatenate([inputs,Up_f01,Up_f11,Up_f02,Up_f12,Up_f03,Up_f13,Up_f04,Up_f14,Up_f05,Up_f15,Up_f06,Up_f16], axis = 3)
    # merge11 = concatenate([Up_f01,Up_f11,Up_f02,Up_f12,Up_f03,Up_f13], axis = 3)
    merge11 = concatenate([Up_f01,Up_f11,Up_f02,Up_f12], axis = 3) # 49

    conv11 = Conv2D(128, 3, activation = 'relu', padding = 'same',)(merge11) # 50

    conv12 = Conv2D(64, 3, activation = 'relu', padding = 'same',)(conv11) # 51
    
    conv13 = Conv2D(64, 3, activation = 'relu', padding = 'same',)(conv12) # 52
    
    conv14 = Conv2D(2, 3, activation = 'tanh', padding = 'same',)(conv13) # 53
    model = Model(inputs, conv14)

    # model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_squared_error', metrics = ['accuracy'])
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    return model

bands = 3
cwd = os.getcwd()
dim=512  # 224
input_shape1 = (dim, dim, 3)
input_shape = (None, None, 1)

model1 = unet1(input_shape)
model1.summary()
print('start***************************************')

model_ref = VGG16(weights='imagenet', include_top=False, input_shape=input_shape1)

w1 =  model_ref.layers[1].get_weights()
w2 = w1[0]
w2m = np.mean(w2,2)
w2m = np.reshape(w2m,(3,3,1,64))
w21 = w1[1]
w1[0] = w2m
w1[1] = w21
# # copy layers of pretrained vgv256 to my model
model1.layers[1].set_weights(w1)
model1.layers[2].set_weights(model_ref.layers[2].get_weights())
model1.layers[4].set_weights(model_ref.layers[4].get_weights())
model1.layers[5].set_weights(model_ref.layers[5].get_weights())
model1.layers[7].set_weights(model_ref.layers[7].get_weights())
model1.layers[8].set_weights(model_ref.layers[8].get_weights())
model1.layers[9].set_weights(model_ref.layers[9].get_weights())
model1.layers[11].set_weights(model_ref.layers[11].get_weights())
model1.layers[12].set_weights(model_ref.layers[12].get_weights())
model1.layers[13].set_weights(model_ref.layers[13].get_weights())
model1.layers[15].set_weights(model_ref.layers[15].get_weights())
model1.layers[16].set_weights(model_ref.layers[16].get_weights())
model1.layers[17].set_weights(model_ref.layers[17].get_weights())


model1.compile(optimizer = Adam(lr = 1e-4), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
 
print('-------------------------------------------------curent path=',cwd)

print('----- read data1 -------')

filesTr_list =  glob.glob(os.path.join(cwd +'/train_images/', '*.png'))


N = len(filesTr_list)
print('------------------------lenth tr = ',str(N))

 
bands = 2
epochs1 = 1
batch_size1 = 4

N1=np.uint16(N/batch_size1) #800
# N1=5
N2=batch_size1

train_imgs=zeros((batch_size1,dim,dim,2))
train_input=zeros((batch_size1,dim,dim,1))

iter0=0

print('start init tr1 --------------------------------------------------------------')
start=datetime.now()
tr_Acc=np.zeros((300,2))
time_Tr=np.zeros((300,2))

MAE_list = []

MAE_min=9999999999.99
stop=0
i=-1
epochs_max=300
max_nb_min=3
nb_min=0
while((i<epochs_max) and (stop==0)):
    i=i+1
    start1=datetime.now()
    iter1=epochs1*(i+1)+iter0
    rand_p=permutation(N)
    idi=0
    lossTr=0
    for i1 in range(50):
        for i2 in range(N2):
            ij=rand_p[idi]
            idi=idi+1
            ima0 = cv2.imread(filesTr_list[ij][:])

            sz0=ima0.shape[0]
            sz1=ima0.shape[1]
            ab=np.zeros((sz0,sz1,2))
            R1 = np.reshape(ima0[:,:,0],(sz0*sz1,1))
            G1 = np.reshape(ima0[:,:,1],(sz0*sz1,1))
            B1 = np.reshape(ima0[:,:,2],(sz0*sz1,1))
            L, A, B = RGB2LAB2(R1,G1,B1)
            A = np.reshape(A,(sz0,sz1))
            B = np.reshape(B,(sz0,sz1))
            ab[:,:,0] = A
            ab[:,:,1] = B
            train_input[i2,:,:,:] =  np.reshape(L,(1,sz0,sz1,1))
            train_imgs[i2,:,:,:] = np.reshape(ab,(1,sz0,sz1,2))
                
        print('HyperUNET iteration number ',(iter1-1),' sub-iter = ',i1+1)
        history_model1 = model1.fit(train_input, train_imgs,
                epochs=epochs1,  # shuffle=True,
                batch_size=batch_size1)
        a=history_model1.history['loss']
        lossTr=lossTr+a[0]
    tr_Acc[iter1-1,0]=iter1-1
    tr_Acc[iter1-1,1]=lossTr/N1
    stopTr=datetime.now()
    timeTr=stopTr-start1
    time_Tr[iter1-1,0]=iter1-1
    time_Tr[iter1-1,1]=timeTr.seconds
    
    if(tr_Acc[iter1-1,1]>MAE_min):
        nb_min=nb_min+1
    else:
        MAE_min = tr_Acc[iter1-1,1]
        nb_min = 0
        model1.save(cwd +'/Hyper_U_NET.h5')
        
    if(nb_min>max_nb_min):
        stop=1          
 
    if(iter1==1):
        model1.compile(optimizer = Adam(lr = 5e-5), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==2):
        model1.compile(optimizer = Adam(lr = 2e-5), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==4):
        model1.compile(optimizer = Adam(lr = 1e-5), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==8):
        model1.compile(optimizer = Adam(lr = 5e-6), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==16):
        model1.compile(optimizer = Adam(lr = 2e-6), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==32):
        model1.compile(optimizer = Adam(lr = 1e-6), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==64):
        model1.compile(optimizer = Adam(lr = 5e-7), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==128):
        model1.compile(optimizer = Adam(lr = 2e-7), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])
    if(iter1==256):
        model1.compile(optimizer = Adam(lr = 1e-7), loss = 'mean_absolute_error', metrics = ['RootMeanSquaredError'])

    np.save(cwd +'/tr_Acc_Hyper_U_NET',tr_Acc)
    np.save(cwd +'/Tr_runtime_Hyper_U_NET',time_Tr)
    

