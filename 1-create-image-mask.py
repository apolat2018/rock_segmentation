"""raster dosyasını ve shape dosyasını açar. Raster dosyayı görüntü olarak kayıt eder.
shape dosyasından mask görüntü oluşturur ve maskelenmiş görüntüyü de kayıt eder."""
import rasterio
import rasterio.mask 
import fiona 
import matplotlib.pyplot as plt
from rasterio.plot import show,reshape_as_raster,reshape_as_image
import cv2
import numpy as np
from skimage import io
import os

data_dir="D:\\rocksegmentation"

raster_file="raster1.tif"
shp_file="rocks.shp"
save_image_file="image.png"
save_mask_file="mask.png"


dataset=rasterio.open(os.path.join(data_dir,raster_file))

image=dataset.read()
print(image.shape)
image=reshape_as_image(image)
#image=cv2.convertScaleAbs(image)#uint16 kayıt yapmıyor uint8 e çevirmek için

image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(data_dir,save_image_file),image)

print(image.shape,image.dtype)

plt.imshow((image))
plt.title("Original image")
plt.show()
#---------------------------------------------------
with rasterio.open(os.path.join(data_dir,raster_file)) as src:
    arr = np.zeros((src.height,src.width)).astype(np.uint8)

    new_dataset = rasterio.open(os.path.join(data_dir,"zeros.tif"), 'w', driver='GTiff',
                            height = arr.shape[0], width = arr.shape[1],
                            count=3, dtype=str(arr.dtype),
                            crs=src.crs,
                            transform=src.transform)

    new_dataset.write(arr, 3)
    new_dataset.close()
#---------------------------------------------------
with fiona.open(os.path.join(data_dir,shp_file), "r") as shapefile:
    shapes = [feature["geometry"] for feature in shapefile]
  
with rasterio.open(os.path.join(data_dir,"zeros.tif")) as src:
    out_image, out_transform = rasterio.mask.mask(src, shapes,filled=True,invert=False,crop=True,nodata=255)
    out_meta = src.meta
    print(src.profile)


img=reshape_as_image(out_image)
#img=reshape_as_image(out_image)
#img=cv2.convertScaleAbs(img)
plt.imshow(img)
plt.title("image with shapes")  
plt.show()

img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
img = np.where(img ==0,255,0) #image renklerine göre değiştir
cv2.imwrite(os.path.join(data_dir,save_mask_file),img)
plt.title("Mask image") 
plt.imshow(img,cmap="gray")   

plt.show()
#-----------------------save-----


print("ALL ıs well")