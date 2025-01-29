import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

# Raster ve shapefile dosyalarını okuyun
raster_path = "D:/rocksegmentation/test_area/dem_p_Clip1.tif"  # Yükseklik verisi
shapefile_path = "D:/rocksegmentation/test_area/predicted_rocks.shp"  # Poligon shapefile

# Shapefile'i yükleyin
gdf = gpd.read_file(shapefile_path)

# İlk geometrinin (poligon) maskelenmesi
polygons = gdf.geometry


with rasterio.open(raster_path) as src:
    dem_data = src.read(1)  # Raster verisi
    transform = src.transform
    nodata_value = src.nodata
    resolution = src.res[0]  # Raster çözünürlüğü (metre cinsinden varsayılır)
    
    # Poligon maskesi oluşturuluyor
    cut_volumes=[]
    fill_volumes=[]
    for n,polygon in enumerate(polygons):
        polygon_mask, polygon_transform = mask(src, [polygon], crop=True)
        
        polygon_mask = polygon_mask[0]

        # Nodata değerlerini minimum yükseklik ile değiştirin
        pol = polygon_mask[polygon_mask != nodata_value]
        
        min_h = np.min(pol)
        polygon_mask = np.where(polygon_mask == nodata_value, min_h, polygon_mask)
        
    

        cut_volume_T=[]
        fill_volume_T=[]
        yukseklikler=[]
        
        row,col=polygon_mask.shape
        for satir in range(row):
            for sutun in range(col):
                if polygon_mask[satir,sutun]!=min_h:
                    
                    yukseklikler.append(polygon_mask[satir,sutun])
            
            #print(len(yukseklikler)) 
            
            if len(yukseklikler)>0:
                h_fark=yukseklikler[-1]-yukseklikler[0]
                if h_fark==0:
                    
                    for e,i in enumerate(yukseklikler):
                        x1=e*resolution
                        
                        if i>yukseklikler[0]:
                            cut_volume=(i-yukseklikler[0])*resolution**2
                            cut_volume_T.append(cut_volume)
                            
                        if i<yukseklikler[0]:
                            fill_volume=(yukseklikler[0]-i)*resolution**2
                            fill_volume_T.append(fill_volume)
                            
                if h_fark>0:
                    oran=(yukseklikler[-1]-yukseklikler[0])/(len(yukseklikler)*resolution)
                    for e,i in enumerate(yukseklikler):
                        x1=oran*(e*resolution)
                        new_h=yukseklikler[0]+x1
                        if new_h<i:
                            cut_volume=(i-new_h)*resolution**2
                            cut_volume_T.append(cut_volume)
                        if new_h>i:
                            fill_volume=(new_h-i)*resolution**2
                            fill_volume_T.append(fill_volume)
                        #print(i,oran,x1,new_h,sum(cut_volume_T))
        
                    
                if h_fark<0:
                    oran=(abs(yukseklikler[-1]-yukseklikler[0]))/(len(yukseklikler)*resolution)
                    for e,i in enumerate(yukseklikler):
                        x1=oran*(e*resolution)
                        new_h=yukseklikler[0]-x1
                        if new_h>i:
                            fill_volume=(new_h-i)*resolution**2
                            fill_volume_T.append(fill_volume)
                        if new_h<i:
                            cut_volume=(i-new_h)*resolution**2
                            cut_volume_T.append(cut_volume)
                        #print(i,oran,x1,new_h,sum(cut_volume_T))

            yukseklikler=[]  
        cut_volumes.append(sum(cut_volume_T))
        fill_volumes.append(sum(fill_volume_T))   
    print(round(sum(cut_volumes),2),round(sum(fill_volumes),2))
    gdf["cut_vol"]= cut_volumes
    gdf["fill_vol"]= fill_volumes
   
    gdf.to_file(shapefile_path)
