import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

raster_path = "D:/rocksegmentation/test_area/dem_p_Clip1.tif"  # Yükseklik verisi
shapefile_path = "D:/rocksegmentation/test_area/predicted_rocks.shp"  # Poligon shapefile


gdf = gpd.read_file(shapefile_path)


polygons = gdf.geometry


with rasterio.open(raster_path) as src:
    dem_data = src.read(1) 
    transform = src.transform
    nodata_value = src.nodata
    resolution = src.res[0]  # Raster çözünürlüğü (metre cinsinden varsayılır)
    
    # Poligon maskesi oluşturuluyor
    cut_volumes=[]
    fill_volumes=[]
    surface_area=[]
    for n,polygon in enumerate(polygons):
        polygon_mask, polygon_transform = mask(src, [polygon], crop=True)
        
        polygon_mask = polygon_mask[0]

        polygon_mask[polygon_mask == nodata_value] = np.nan

        grad_y, grad_x = np.gradient(polygon_mask, resolution, resolution)

        # 3D alan hesaplama
        slope_correction = np.sqrt(1 + grad_x**2 + grad_y**2)
        pixel_area=resolution*resolution
        pixel_3d_area = pixel_area * slope_correction
        total_3d_area = np.nansum(pixel_3d_area)  
        surface_area.append(total_3d_area)

    print(sum(surface_area))    
    gdf["3D_area"]=surface_area
    gdf.to_file(shapefile_path)