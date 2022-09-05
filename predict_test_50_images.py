# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:00:19 2021

@author: salim
"""

import skimage
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
    
    # Y=0.3*R + 0.59*G + 0.11*B
    # X=0.45*R + 0.35*G + 0.2*B
    # Z=0.01*R + 0.09*G + 0.9*B
    
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
    # C = np.array([L, a, b])
    # print(C.shape)
    # print(L.shape)
    # print(a.shape)
    # print(b.shape)
    # print(aa.shape)
    
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
    # p0=np.where(L<0.02)
    # R[p0[0]]=0   
    # G[p0[0]]=0
    # B[p0[0]]=0
    # p1=np.where(L>0.98)
    # R[p1[0]]=255   
    # G[p1[0]]=255
    # B[p1[0]]=255
    return R, G, B


def psnr(img1, img2):
    mse = np.mean( (img1.astype("float") - img2.astype("float")) ** 2 )
    # print(mse)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def mse(imageA, imageB, nband):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * nband)
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def mae(imageA, imageB, bands):
 	# the 'Mean Squared Error' between the two images is the
 	# sum of the squared difference between the two images;
 	# NOTE: the two images must have the same dimension
 	err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
 	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
    
def rmse(imageA, imageB, nband):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1] * nband)
	err = np.sqrt(err)
	return err
    

bands = 2
cwd = os.getcwd()
dim=512  # 224

model1=load_model(cwd +'/Hyper_U_Net.h5')



print('----- read data1 -------')

filesTs_list =  glob.glob(os.path.join(cwd +'\\test_50images\\', '*.jpg'))
files_name1 = [fn.split('\\')[-1].split('.jpg')[0].strip() for fn in filesTs_list]

N = len(filesTs_list)
print('------------------------lenth ts = ',str(N))
    



MSE_list = []
MAE_list = []
MSEr_list = []
MSEg_list = []
MSEb_list = []
RMSE_list = []
PSNR_list = []
SSIM_list = []
deltaE_list = []

for j1 in range(N):
    print('final2 image = ',j1)
    # ima0 = img_to_array(load_img(filesTs_list[j1][:]))
    # if(ima0.shape[0]%64>0):
    #     d00 = int(ima0.shape[0]/64)
    #     d0 = (d00+1)*64
    # else:
    #     d0=ima0.shape[0]
    # if(ima0.shape[1]%64>0):
    #     d11 = int(ima0.shape[1]/64)
    #     d1 = (d11+1)*64
    # else:
    #     d1=ima0.shape[1]
                    
    # ima = np.zeros((d0,d1,ima0.shape[2]))
    # ima[0:ima0.shape[0],0:ima0.shape[1],:] = ima0
    # else:
                #     ima = ima0
                
                
    ima = cv2.imread(filesTs_list[j1][:])
    ima0 = ima

                
    sz0 = ima.shape[0]
    sz1 = ima.shape[1]
    sz2 = bands #ima.shape[2]
                    
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
    ima_gray =  np.reshape(L,(1,sz0,sz1,1))                    
                
    predicted = model1.predict(ima_gray,verbose=0)
    
    predicted = np.reshape(predicted,(sz0*sz1,bands))
    Ar = predicted[:,0:1]
    Br = predicted[:,1:2]

    Rr, Gr, Br = LAB22RGB(L,Ar,Br)
    Rr = np.reshape(Rr,(sz0,sz1))
    Gr = np.reshape(Gr,(sz0,sz1))
    Br = np.reshape(Br,(sz0,sz1))
    predicted255=np.uint8(np.zeros((sz0,sz1,3)))
    predicted255[:,:,0] = Rr
    predicted255[:,:,1] = Gr
    predicted255[:,:,2] = Br
    
    cv2.imwrite(cwd +'/pred_test_50images_HyperUNet/'+files_name1[j1][:]+'.png',predicted255)


        
                
    MSE = mse(ima0,predicted255,3)
    MSEr = mse(ima0[:,:,0],predicted255[:,:,0],1)
    MSEg = mse(ima0[:,:,1],predicted255[:,:,1],1)
    MSEb = mse(ima0[:,:,2],predicted255[:,:,2],1)
    RMSE = rmse(ima0,predicted255,3) #
    # MAE = mae(ima0,predicted255,3)
    err = np.sum(np.abs(ima0.astype("float") - predicted255.astype("float")))
    MAE = err/float(ima0.shape[0] * predicted255.shape[1] * 3)    
    # print('MAE = ',MAE)
        # PSNR = tf.image.psnr(, predicted255 , max_val=255)
    PSNR=psnr(ima0,predicted255)
    SSIM = ssim(ima0,predicted255, multichannel=True)
    
    ima0 = np.float32(ima0)
    predicted255 = np.float32(predicted255)
    ima0 *= 1./255
    predicted255 *= 1./255
    Lab1 = cv2.cvtColor(ima0, cv2.COLOR_BGR2Lab)
    Lab2 = cv2.cvtColor(predicted255, cv2.COLOR_BGR2Lab)
    L1, a1, b1 = cv2.split(Lab1)
    L2, a2, b2 = cv2.split(Lab2)
    #print(L2)
    
    Kl=1
    KC=1
    KH=1
    
    delta=skimage.color.deltaE_ciede2000(L1,L2, Kl, KC, KH)
    #print(len(delta))
    
    deltaE=np.mean(delta)
    # print("deltaE", m)
    
    # deltaE=delta_e_cie2000(ima0, predicted255, Kl=1, Kc=1, Kh=1)
                
    MSE_list.append(MSE)
    MSEr_list.append(MSEr)
    MSEg_list.append(MSEg)
    MSEb_list.append(MSEb)
    RMSE_list.append(RMSE)
    MAE_list.append(MAE)
    deltaE_list.append(deltaE)
        # MSE2_list.append(MSE2)
        # MAE2_list.append(MAE2)    
    SSIM_list.append(SSIM)
    PSNR_list.append(PSNR)

    # id_Ac = id_Ac + 1

MSE_mean = np.mean(MSE_list)
MAE_mean = np.mean(MAE_list)
PSNR_mean = np.mean(PSNR_list)
SSIM_mean = np.mean(SSIM_list)
MSEr_mean = np.mean(MSEr_list)
MSEg_mean = np.mean(MSEg_list)
MSEb_mean = np.mean(MSEb_list)
RMSE_mean = np.mean(RMSE_list)
deltaE_mean = np.mean(deltaE_list)

# print('MSE_mean=',MSE_mean)
# print('RMSE_mean=',RMSE_mean)
print('deltaE_mean=',deltaE_mean)
print('MAE_mean=',MAE_mean)
print('PSNR_mean=',PSNR_mean)
print('SSIM_mean=',SSIM_mean*100)


