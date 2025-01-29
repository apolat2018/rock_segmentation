# -*- coding: utf-8 -*-

#180. satır image size göre değiştirilmelidir. Bu kod 128x128 için yazılmıştır.
import os,sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import segmentation_models as sm
import cv2
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import albumentations as A
import geopandas as gpd
import time
from skimage import io
from tqdm import tqdm
sm.set_framework('tf.keras')
sm.framework()
print("sm: ",sm.__version__,"keras:",keras.__version__,"tf: ",tf.__version__)


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(18, 6))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
    
# helper function for data visualization    
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x
def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)

image_path="D:/rocksegmentation/test_area/test_ras1.tif"
c=cv2.imread(image_path)
plt.imshow(c)
plt.show()
print(c.shape)

BACKBONE = 'densenet121'
#'densenet121'fenadeğil
#'resnet101'
#'resnet34'
#'efficientnetb6'
#'inceptionresnetv2'
preprocess_input = sm.get_preprocessing(BACKBONE)
#create model
model = sm.Unet(BACKBONE, activation="sigmoid")

# load best weights
model.load_weights("D:\\rocksegmentation\\codes\\densenet121result\\best_model_tflite.h5")

#---------------------------------make predicted image -------------------------------------------------------------------
save_folder="D:/rocksegmentation/test_area"

h=256
w=256

def create_image(image_path,save_folder,h,w):
    img=cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    back=np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8)
    thickness=3#Karelaj kalınlık
    color=(255,0,0)#karelaj renk
    #alan=img.shape[0]*img.shape[1]
    #print(img.shape)

    h=h
    w=w
    #img2=np.zeros([h,w,3],dtype=np.float32)
    sutun=int(img.shape[1]/w)
    satir=int(img.shape[0]/h)
    print("satir sayısı= ",satir,"sütün sayısı= ",sutun)
    x1=0
    y1=0
    x2=w
    y2=h

    for j in tqdm(range(satir)):
        #print("satir= ",j)
        
       
        for i in range(sutun):
            #print("y1, y2",y1,y2,"-------------""x1,x2",x1,x2)
            if x2>img.shape[1]:
                print(img.shape[1])
                x1=0
                x2=w
            #print("Sütun= ",i)
            #print("buaraya geliyorum")
            ROI=img[y1:y2,x1:x2]
            #name=os.path.join(save_folder,"file"+str(j)+str(i)+".jpg")
            part=preprocess_input(ROI)
            part=np.expand_dims(part,axis=0)
            pred=model.predict(part)>0.5
            pred=pred[...,0].squeeze()
            
            #pred=pred#buna bakalım
            
            #print(y1,y2,x1,x2)
            back[y1:y2,x1:x2]=pred
            #ROI = cv2.putText(ROI, str(j)+str(i)+"ROI", (10,100), font,1, (0,0,255), 2, cv2.LINE_AA)
            #cv2.imwrite(name,ROI)
            #cv2.rectangle(back,(x1,y1),(x2,y2),color=color,thickness=thickness)
            #cv2.putText(back, str(j)+str(i),(x1,y1), font,3, (0,255,0), 5, cv2.LINE_AA)
            
                       
            #x1=x1+w
            #x2=x1+w
            x1=x1+w
            x2=x1+w
            

            
            
        #y1=y1+h
        #y2=y1+h
        y1=y1+h
        y2=y1+h
    plt.imshow(back)
    plt.show()
    cv2.imwrite(os.path.join(save_folder,"result.png"),back)      
    print("Predicted image was created")
        
        
create_image(image_path,save_folder,h,w)

        
#2.turda kenarlardaki bozuklukları gidermek için             
def create_image_2(image_path,save_folder,h,w,ofset):
    img=cv2.imread(image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    rslt=cv2.imread(os.path.join(save_folder,"result.png"),0)#1. aşamada tahmin edilenler
    h=h
    w=w
    #img2=np.zeros([h,w,3],dtype=np.float32)
    sutun=int(img.shape[1]/w)
    satir=int(img.shape[0]/h)
    #print("satir sayısı= ",satir,"sütün sayısı= ",sutun)
    x1=0
    y1=0
    x2=w
    y2=h

    for j in tqdm(range(satir+1)):
        #print("satir= ",j)
        
       
        for i in range(sutun+1):
            #print("y1, y2",y1,y2,"-------------""x1,x2",x1,x2)
            if x2>img.shape[1]:
                print(img.shape[1])
                x1=0
                x2=w
            #print("Sütun= ",i)
            #print("buaraya geliyorum")
            ROI=img[y1:y2,x1:x2]
            #name=os.path.join(save_folder,"file"+str(j)+str(i)+".jpg")
            part=preprocess_input(ROI)
            part=np.expand_dims(part,axis=0)
            pred=model.predict(part)>0.5
            pred=pred[...,0].squeeze()
            
            #pred=pred#buna bakalım
            
            #print(y1,y2,x1,x2)
            #rslt[y1:y2-50,x1+50:x2-50]=pred[50:78,50:78]
            rslt[y1:y2,x1+ofset:x2-ofset]=pred[0:h,ofset:w-ofset]
            #ROI = cv2.putText(ROI, str(j)+str(i)+"ROI", (10,100), font,1, (0,0,255), 2, cv2.LINE_AA)
            #cv2.imwrite(name,ROI)
            #cv2.rectangle(back,(x1,y1),(x2,y2),color=color,thickness=thickness)
            #cv2.putText(back, str(j)+str(i),(x1,y1), font,3, (0,255,0), 5, cv2.LINE_AA)
            
                       
            #x1=x1+w
            #x2=x1+w
            x1=x2-ofset#overlay 50px
            x2=x1+w
            #print(x1)

            
            
        #y1=y1+h
        #y2=y1+h
        y1=y1+h-ofset#overlay 50px
        y2=y1+h
            
            
             
    plt.imshow(rslt)
    plt.show()
    cv2.imwrite(os.path.join(save_folder,"result_son.png"),rslt)      
    print("Predicted image was created")


create_image_2(image_path,save_folder,h,w,64)
print(np.unique(cv2.imread(os.path.join(save_folder,"result.png"))))

#50px overlay kısmına bakılacak

