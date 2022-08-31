# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 08:00:19 2021

@author: salim
"""


# from model1 import *
from datetime import datetime
# from utils import utils, helpers
import PIL
from PIL import Image, ImageOps
from scipy import ndimage
import math
import glob
import cv2
import numpy as np
from numpy import zeros
from numpy import ones
from numpy import vstack, hstack
from numpy.random import randn, rand
from numpy.random import randint, permutation
import random
import os
# from tkinter import *
# from tkinter.filedialog import askdirectory
# from tkinter import filedialog
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models
from sklearn.utils import shuffle
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np 
import os
from skimage.color import rgb2lab, lab2rgb
import cv2
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from PIL import Image
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

# model1=load_model(cwd +'/Unet1_subImageNet3_SP_7it.h5') # 1 initial trained on Imagenet3

# model1=load_model(cwd +'/HyperColumn1_2_ImageNet3_SP_3it.h5') # 2 tuned using the initial with imagenet merge 2+2
# model1=load_model(cwd +'/HyperColumn2_2_ImageNet3_SP_5it.h5') # 3 tuned using the initial with imagenet merge 3+3
# model1=load_model(cwd +'/HyperColumn3_2_ImageNet3_8it.h5') # 4 tuned using the initial with imagenet merge 4+4

# model1=load_model(cwd +'/HyperColumn1_2_Full_ImageNet3_5it.h5') # 5 trained from scratch with imagenet merge 2+2
# model1=load_model(cwd +'/HyperColumn2_2_Full_ImageNet3_SP_2it.h5') # 6 trained from scratch with imagenet merge 3+3
# model1=load_model(cwd +'/HyperColumn3_2_Full_ImageNet3_6it.h5') # 7 trained from scratch with imagenet merge 4+4

# model1=load_model(cwd +'/Unet1_SUN64_subImageNet3_SP_15it.h5') # 8: 1 tuned on SUN64

# model1=load_model(cwd +'/HyperColumn01_2_SUN64_ImageNet3_5it.h5') # 9: 2 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn1_2_SUN64_ImageNet3_SP_6it.h5') # 9: 2 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn2_2_SUN64_ImageNet3_SP_6it.h5') # 10: 3 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn3_2_SUN64_ImageNet3_SP_8it.h5') # 11: 4 tuned on SUN64

# model1=load_model(cwd +'/HyperColumn3_2_Full_ImageNet3_6it.h5') # 12: 5 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn2_2_Full_SUN64_ImageNet3_SP_5it.h5') # 13: 6 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn3_2_Full_SUN64_ImageNet3_SP_6it.h5') # 14: 7 tuned on SUN64

# model1=load_model(cwd +'/HyperColumn1_2_SUN64_ImageNet3_SP_rgb_10it.h5') # 9: 2 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn2_2_SUN64_ImageNet3_SP_rgb_8it.h5') # 10: 3 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn3_2_SUN64_ImageNet3_SP_rgb_9it.h5') # 11: 4 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn1_2_AERAL_IM_3it.h5') # 11: 4 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn_final3_Norm_10it.h5') # 11: 4 tuned on SUN64
# model1=load_model(cwd +'/HyperColumn1_2_LAB_Norm4_tanh_100it.h5')
# model1=load_model(cwd +'/HyperColumn1_2_LAB_Norm1_20it.h5')
# model1=load_model(cwd +'/HyperColumn1_2_LAB_Norm2_tanh_100it.h5')
model1=load_model(cwd +'/HyperColumn1_2_LAB22_Norm4_tanh_32it.h5')

# Norm = 0   # for tanh3
Norm = 1   # for tanh1 and tanh2 and tanh4


print('----- read data1 -------')

# filesTs_list =  glob.glob(os.path.join(cwd +'/historical_aerial_bw/24ago44/', '*.png'))
# files_name1 = [fn.split('/')[-1].split('.png')[0].strip() for fn in filesTs_list]

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
# N=1
# ima0 = img_to_array(load_img('C:/FBK/colorisation/ex_7_im/ex_images/output/22_59_19_2_2.png'))
# predicted2550= img_to_array(load_img('C:/FBK/colorisation/ex_7_im/ex_images/output/22_59_19_2_2.png'))
for j1 in range(1):
# for j1 in range(N):
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
                
                
    ima = cv2.imread('C:/work/colorisation/compare_histo/2.jpg')
    ima0 = ima

                
    # sz0 = ima.shape[0]
    # sz1 = ima.shape[1]
    # sz2 = bands #ima.shape[2]
    #                 # ima_gray = np.mean(ima,2)
                    
    # L0 = ima0[:,:,0]
    # L=L0/255

    # ima_gray =  np.reshape(L,(1,sz0,sz1,1))                    
                    
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
    ima_gray =  np.reshape(L,(sz0,sz1,1))
    ima_gray=np.uint8(np.round(L*255))
    cv2.imwrite('C:/work/colorisation/compare_histo/2_gray.jpg',ima_gray)
    ima_gray =  np.reshape(L,(1,sz0,sz1,1))
                    
                
    # ima_gray = np.mean(ima,2)
    # ima_LAB = rgb2lab(ima)
    
        # imaPIL0 = Image.fromarray(np.uint8(ima0))
        # ima256_gray0 = ImageOps.grayscale(imaPIL0)
        # ima_gray0  = img_to_array(ima256_gray0) 
                   
        # imaPIL = Image.fromarray(np.uint8(ima))
        # ima256_gray = ImageOps.grayscale(imaPIL)
        # ima_gray  = img_to_array(ima256_gray)                
    # ima_gray = ima_gray/255 
    # ima_gray = np.reshape(ima_gray,(1,sz0,sz1,1))
    # ima_gray =  np.reshape(ima[:,:,0],(1,ima.shape[0],ima.shape[1],1))
    # if(Norm==1):
    #     # max1=1
    #     ima_gray=ima_gray/255
    # else:
        # max1=255
    predicted = model1.predict(ima_gray,verbose=0)
    
    predicted = np.reshape(predicted,(sz0*sz1,bands))
    predicted = np.reshape(predicted,(sz0*sz1,bands))
    a0 = predicted[:,0:1]
    b0 = predicted[:,1:2]
    # if(Norm==1):
    # predicted = predicted*127
    # else:
        # predicted = predicted - 127
        
    # predictedLAB = np.zeros((sz0,sz1,3))    
    # predictedLAB[:,:,0] =  ima_LAB[:,:,0]
    # predictedLAB[:,:,1:3] =  predicted
    Lr=np.reshape(L,(sz0*sz1,1))
    
    Rr, Gr, Br = LAB22RGB(Lr,a0,b0)
    Rr = np.reshape(Rr,(sz0,sz1))
    Gr = np.reshape(Gr,(sz0,sz1))
    Br = np.reshape(Br,(sz0,sz1))
    predicted255=np.uint8(np.zeros((sz0,sz1,3)))
    predicted255[:,:,0] = Rr
    predicted255[:,:,1] = Gr
    predicted255[:,:,2] = Br
    
    # print('min=',predicted.min())
    # print('max=',predicted.max())
    # predicted1D = np.reshape(predicted,(sz0*sz1*bands,1))
    # a = np.where(predicted1D>max1)
    # predicted1D[a[0]]=max1
    # print('*** number of sub = ',len(a[0]))
    # print('len a = ',len(a[0]))
    # predicted = np.reshape(predicted1D,(sz0,sz1,bands))
    # predicted = np.reshape(predicted,(sz0,sz1,bands))
    # # if(Norm==1):
    # predicted = predicted*127
    # # else:
    #     # predicted = predicted - 127
        
    # predictedLAB = np.zeros((sz0,sz1,3))    
    # predictedLAB[:,:,0] =  ima[:,:,0]
    # predictedLAB[:,:,1:3] =  predicted
    # predicted255 = lab2rgb(predictedLAB)
    # predicted255 = np.uint8(np.round(predicted255*255))
    # if(Norm==1):
    #     cv2.imwrite(cwd +'/aerial_images_01/pred_HyperColumn1_2_LAB_Norm1_20it/'+files_name1[j1][:]+'.jpg',predicted255)
    # else:
    #     cv2.imwrite(cwd +'/aerial_images_01/pred_HyperColumn1_2_LAB_1_20it/'+files_name1[j1][:]+'.jpg',predicted255)
    # predicted[:,:,0]=3*ima_gray[0,:,:,0]-predicted[:,:,1] - predicted[:,:,2]
        # predicted2550 = predicted[0:ima0.shape[0],0:ima0.shape[1],:]*255
       
    # predicted255 = np.uint8(np.round(predicted[0:ima0.shape[0],0:ima0.shape[1],:]*255))
    # predicted255 = np.uint8(predicted[0:ima0.shape[0],0:ima0.shape[1],:])
    # predicted255 = np.uint8(np.round(predicted[0:ima0.shape[0],0:ima0.shape[1],:]*255))
    # print(cwd)
    print('C:/work/colorisation/compare_histo/2_res.jpg',predicted255)
        
        # predicted2550[:,:,2:3] = (ima_gray0 - 0.299*predicted2550[:,:,0:1] - 0.587*predicted2550[:,:,1:2])/0.114


        # predicted1D = np.reshape(predicted2550,(ima0.shape[0]*ima0.shape[1]*bands,1))
        
        # a = np.where(predicted1D>255)
        # predicted1D[a[0]]=255
        # a = np.where(predicted1D<0)
        # predicted1D[a[0]]=0
                
#     MSE = mse(ima0,predicted255,3)
#     MSEr = mse(ima0[:,:,0],predicted255[:,:,0],1)
#     MSEg = mse(ima0[:,:,1],predicted255[:,:,1],1)
#     MSEb = mse(ima0[:,:,2],predicted255[:,:,2],1)
#     RMSE = rmse(ima0,predicted255,3) # mae(imagt255,predicted255,bands)
#         # PSNR = tf.image.psnr(, predicted255 , max_val=255)
#     PSNR=psnr(ima0,predicted255)
#     SSIM = ssim(ima0,predicted255, multichannel=True)
                
#     MSE_list.append(MSE)
#     MSEr_list.append(MSEr)
#     MSEg_list.append(MSEg)
#     MSEb_list.append(MSEb)
#     RMSE_list.append(RMSE)
#         # MSE2_list.append(MSE2)
#         # MAE2_list.append(MAE2)    
#     SSIM_list.append(SSIM)
#     PSNR_list.append(PSNR)

#     # id_Ac = id_Ac + 1

# MSE_mean = np.mean(MSE_list)
# PSNR_mean = np.mean(PSNR_list)
# SSIM_mean = np.mean(SSIM_list)
# MSEr_mean = np.mean(MSEr_list)
# MSEg_mean = np.mean(MSEg_list)
# MSEb_mean = np.mean(MSEb_list)
# RMSE_mean = np.mean(RMSE_list)

# print('MSE_mean=',MSE_mean)
# print('RMSE_mean=',RMSE_mean)

# print('PSNR_mean=',PSNR_mean)
# print('SSIM_mean=',SSIM_mean*100)

# print('MSEr_mean=',MSEr_mean)
# print('MSEg_mean=',MSEg_mean)
# print('MSEb_mean=',MSEb_mean)

