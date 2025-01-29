# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 14:07:20 2021
V1.0 Bu versionda train ve test görüntülerinin aynı pikselleri içerme ihtimali çok yüksek olduğundan V1.1 yazılacak
@author: ap
"""
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import shutil

image_path="D:\\rocksegmentation\\image.png"
mask_path="D:\\rocksegmentation\\mask.png"
dimension=256
image_number=560
split_size=0.66#% train test split
save_test_image_path=f"D:\\rocksegmentation\\random_sampling_dataset_{dimension}\\val"
save_test_mask_path=f"D:\\rocksegmentation\\random_sampling_dataset_{dimension}\\val_mask"
save_train_image_path=f"D:\\rocksegmentation\\random_sampling_dataset_{dimension}\\train"
save_train_mask_path=f"D:\\rocksegmentation\\random_sampling_dataset_{dimension}\\train_mask"

def create_folder(folder):
    if os.path.exists(folder):
        print(f"{folder} zaten var")
        shutil.rmtree(folder)
    else:
        os.makedirs(folder)
create_folder(save_test_image_path)
create_folder(save_test_mask_path)
create_folder(save_train_image_path)
create_folder(save_train_mask_path)
   
img=cv2.imread(image_path)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
mask=cv2.imread(mask_path)
"""plt.imshow(img)
plt.show()"""


train_number=int(image_number*split_size)
test_number=image_number-train_number
height=img.shape[0]
weight=img.shape[1]
h=int(height/3)
w=int(weight/3)

x1=0
y1=0
x2=w
y2=h
n=0#1. 5. ve 9. görüntüler test image için diğerleri train image için 

def random_sampling(save_image_path,save_mask_path,image_number):
     
    X=[]
    Y=[]
    max_X=img.shape[1]
    max_Y=img.shape[0]
    s=0
    while s<image_number:
        
        Rx=random.randint(0, max_X-dimension)
        if Rx in X:
            pass
        else:
            X.append(Rx)
            Ry=random.randint(0, max_Y-dimension)
            if Ry in Y:
                pass
            else:
                Y.append(Ry)
                
                ROI_img=img[Ry:Ry+dimension,Rx:Rx+dimension]
                ROI_img=cv2.cvtColor(ROI_img,cv2.COLOR_RGB2BGR)
                ROI_mask=mask[Ry:Ry+dimension,Rx:Rx+dimension]
                #ROI_mask=cv2.cvtColor(ROI_mask,cv2.COLOR_RGB2BGR)
                toplam=ROI_mask.shape[0]*ROI_mask.shape[1]
                a=np.count_nonzero(ROI_mask)
                oran=(a/toplam)*100
               
                if oran<1 or ROI_img.shape[0]!=dimension or ROI_img.shape[1]!=dimension or ROI_mask.shape[0]!=dimension or ROI_mask.shape[1]!=dimension:
                    pass
                

                else:
                    s=s+1
                    if os.path.isdir(save_image_path)==False:
                        os.mkdir(save_image_path)
                    if os.path.isdir(save_mask_path)==False:
                        os.mkdir(save_mask_path)
                        
                        
                    cv2.imwrite(os.path.join(save_image_path, str(s)+".png"),ROI_img)
                   
                    cv2.imwrite(os.path.join(save_mask_path, str(s)+".png"),ROI_mask)
                    print(s)
                    print("X syısı",len(X))
                    
                    
                    
for j in tqdm(range(3)):
    #print("satir= ",j)
  
    for i in range(3):
        #print("Sütun= ",i)
        
    

        ROI_img=img[y1:y2,x1:x2]
        ROI_mask=mask[y1:y2,x1:x2]
        #name=os.path.join(save_folder,set_name+str(j)+"_"+str(i)+".png")
        
        ROI_img=cv2.cvtColor(ROI_img,cv2.COLOR_RGB2BGR)
        """plt.subplot(121)
        plt.imshow(ROI_img)
        plt.subplot(122)
        plt.imshow(ROI_mask)
        plt.show()"""
        #cv2.imwrite(name,ROI)
        #create grid
        #cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),10)
        x1=x1+w
        x2=x1+w
        n=n+1
        if n==1 or n==3 or n==8:
            random_sampling(save_test_image_path, save_test_mask_path, test_number)
        else:
            random_sampling(save_train_image_path, save_train_mask_path, train_number)
            
        #print("n=",n)
        if x2>img.shape[1]:
            x1=0
            x2=w
            y1=y1+h
            y2=y1+h






























