# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 15:09:44 2021

@author: ap
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon,Point,MultiPolygon
from fiona.crs import from_epsg
import rasterio

mask=cv2.imread("D:/rocksegmentation/test_area/result_son.png",0)
raster="D:/rocksegmentation/test_area/test_ras1.tif"


with rasterio.open(raster) as src:
    t=src.transform
    CRS=src.crs
    print(CRS)

offset=128#orijinal görüntü daha büyük bir görüntütünün içine yerleştirilecek x ve y ye eklenecek
bosluk=int(offset/2)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))


new_img=np.zeros((mask.shape[0]+offset,mask.shape[1]+offset),np.uint8)
new_img[bosluk:new_img.shape[0]-bosluk,bosluk:new_img.shape[1]-bosluk]=mask
plt.imshow(new_img)
plt.show()

im = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)#gürültüleri temizlemek için
#im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)#gürültüleri temizlemek için
im = cv2.morphologyEx(im, cv2.MORPH_ELLIPSE, kernel)#gürültüleri temizlemek için

#im = cv2.GaussianBlur(mask, (7, 7), 0)

#edge = cv2.Canny(im, 0, 1)

contours,_ = cv2.findContours(im.copy(), 
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

"""im_cnt=cv2.drawContours(im, contours, -1, (255,0,0), thickness = 1)

im_cnt=np.where(im_cnt>0,255,0)
plt.imshow(im_cnt[200:300,0:80])
plt.show()
cv2.imwrite("D:/rocksegmentation/predict_data/deneme.png",im_cnt)"""

"""contours_list=list(contours)

for i,cor in enumerate(contours_list):
    a=cv2.contourArea(cor)
    
    if a<15:
        print(i,a)
        del contours_list[i]
        
        
contours=tuple(contours_list)"""
        
    

new_coor=[]
for cnt in contours:
    cnt=np.squeeze(cnt)
    #print(cv2.contourArea(cnt))
    n=[]
    for i in range(len(cnt)):
       
        x=0.
        y=0.
        #cnt[i][0]=float(t[2]+(cnt[i][0]*t[0])) #left coordinate *cell size
        x=float((t[2])+((cnt[i][0]-bosluk)*t[0]))
       
        #cnt[i][1]=float(t[5]-(cnt[i][1]*-(t[4]))) 
        y=float((t[5])-((cnt[i][1]-bosluk)*-(t[4])))#Up coordinate*cell size
     
      
        a=[x,y]
        n.append(a)
        #print("after",cnt[i][0])
    new_coor.append(n)
       
    
        
#cv2.drawContours(im, contours, -1, (0,255,0), thickness = 2)
#rslts=im.copy()
#rslts=np.zeros((512,512,3),dtype=np.float32)

#cv2.polylines(image,contours,True,(0,255,0),3)


"""fp="D:/rocksegmentation/rocks.shp"
data=gpd.read_file(fp)
#data.plot()
#plt.show()
print(data.crs)
print(data.head(5))
for h in contours:
    A=cv2.contourArea(h)"""

new=gpd.GeoDataFrame()
alan=[]
for i,poly in enumerate(new_coor):
    
   
   
    p=Polygon(np.squeeze(poly))
    alan.append(p.area)
    #if p.area>0.1:#çok küçük poligonları iptal etmek için. Burayı düşünelim
    new.loc[i,"geometry"]=p
  


new.crs=CRS

#print(new.head(5),new.crs)#projection
new.to_file("D:/rocksegmentation/test_area/predicted_rocks.shp",)


new.plot()
plt.show()