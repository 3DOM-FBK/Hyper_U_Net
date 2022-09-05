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

# def mae(imageA, imageB, bands):
# 	# the 'Mean Squared Error' between the two images is the
# 	# sum of the squared difference between the two images;
# 	# NOTE: the two images must have the same dimension
# 	err = np.sum(np.abs(imageA.astype("float") - imageB.astype("float")))
# 	err /= float(imageA.shape[0] * imageA.shape[1] * bands)
    
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


# Norm = 0   # for tanh3
Norm = 1   # for tanh1 and tanh2 and tanh4


print('----- read data1 -------')

filesTs_list =  glob.glob(os.path.join(cwd +'\\Histo_images\\', '*.png'))
files_name1 = [fn.split('\\')[-1].split('.png')[0].strip() for fn in filesTs_list]

N = len(filesTs_list)
# Nval = len(filesVal_list)
# N = N1 * N2
print('------------------------lenth ts = ',str(N))
    



MSE_list = []
MSEr_list = []
MSEg_list = []
MSEb_list = []
RMSE_list = []
PSNR_list = []
SSIM_list = []

for j1 in range(N):
    print('final2 image = ',j1)
    print(files_name1[j1][:])
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
                    # ima_gray = np.mean(ima,2)
                    
    L0 = ima0[:,:,0]
    L=L0/255

    ima_gray =  np.reshape(L,(1,sz0,sz1,1))                    
                    
    predicted = model1.predict(ima_gray,verbose=0)
    
    predicted = np.reshape(predicted,(sz0*sz1,bands))
    predicted = np.reshape(predicted,(sz0*sz1,bands))
    a0 = predicted[:,0:1]
    b0 = predicted[:,1:2]

    Lr=np.reshape(L,(sz0*sz1,1))
    
    Rr, Gr, Br = LAB22RGB(Lr,a0,b0)
    Rr = np.reshape(Rr,(sz0,sz1))
    Gr = np.reshape(Gr,(sz0,sz1))
    Br = np.reshape(Br,(sz0,sz1))
    predicted255=np.uint8(np.zeros((sz0,sz1,3)))
    predicted255[:,:,0] = Rr
    predicted255[:,:,1] = Gr
    predicted255[:,:,2] = Br

    cv2.imwrite(cwd +'/pred_Histo_images/'+files_name1[j1][:]+'.png',predicted255)
        