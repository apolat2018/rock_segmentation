import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio import features
from shapely.geometry import mapping
import os
raster_path = r"C:/AP/rock_seg/veri/dem_p_Clip1.tif"
shapefile_path = r"C:/AP/rock_seg/veri/predicted_rocks_10buf.shp"

gdf = gpd.read_file(shapefile_path)

def fit_plane_least_squares(x, y, z):
    """Fit plane z = a*x + b*y + c using least squares."""
    A = np.c_[x, y, np.ones_like(x)]
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coef
    return float(a), float(b), float(c)

def xy_grids_from_transform(transform, height, width):
    """Create 2D X,Y grids (cell centers) for a window transform."""
    rows = np.arange(height)
    cols = np.arange(width)
    cc, rr = np.meshgrid(cols, rows)
    X = transform.c + (cc + 0.5) * transform.a + (rr + 0.5) * transform.b
    Y = transform.f + (cc + 0.5) * transform.d + (rr + 0.5) * transform.e
    return X.astype("float64"), Y.astype("float64")

def pixel_area_from_transform(transform):
    px_w = transform.a
    px_h = abs(transform.e)
    return float(px_w * px_h), float(px_w), float(px_h)

def rasterize_bool(geom, out_shape, transform, all_touched=False):
    """Rasterize geometry into a boolean mask."""
    return features.rasterize(
        [(mapping(geom), 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=all_touched
    ).astype(bool)

results = []

with rasterio.open(raster_path) as src:
    dem_nodata = src.nodata
    dem_crs = src.crs

    # CRS kontrolü (hızlı güvenlik)
    if gdf.crs is None:
        raise ValueError("Shapefile CRS tanımsız. (gdf.crs None) Lütfen CRS ata.")
    if dem_crs is None:
        raise ValueError("DEM CRS tanımsız. (src.crs None)")
    if gdf.crs != dem_crs:
        gdf = gdf.to_crs(dem_crs)

    for i, polygon in enumerate(gdf.geometry):
        # 1) Poligon alanını kırp
        out_img, out_transform = mask(src, [polygon], crop=True)
        dem_crop = out_img[0].astype("float64")  # (H,W)

        # 2) Nodata temizle (senin yaptığın gibi minimum ile doldurma)
        #    Not: Bu, hacmi etkileyebilir. İstersen NaN bırakıp dışarıda tutabiliriz.
        if dem_nodata is not None:
            dem_crop = np.where(dem_crop == dem_nodata, np.nan, dem_crop)

        # 3) Crop grid için X,Y üret
        H, W = dem_crop.shape
        X, Y = xy_grids_from_transform(out_transform, H, W)
        pixel_area, px_w, px_h = pixel_area_from_transform(out_transform)

        # 4) Crop içinde poligon içi maskesi
        inside = rasterize_bool(polygon, (H, W), out_transform, all_touched=False)

        # 5) Sadece poligon sınırından (boundary) örnek al
        boundary = polygon.boundary
        edge = rasterize_bool(boundary, (H, W), out_transform, all_touched=True)

        zb = dem_crop[edge]
        xb = X[edge]
        yb = Y[edge]

        v = np.isfinite(zb) & np.isfinite(xb) & np.isfinite(yb)
        zb, xb, yb = zb[v], xb[v], yb[v]

        if zb.size < 15:
            results.append({
                "polygon_id": i,
                "error": f"Not enough boundary samples: {zb.size}"
            })
            continue

        # 6) Best-fit plane: z = a*x + b*y + c
        a, b, c = fit_plane_least_squares(xb, yb, zb)

        # 7) Δh ve cut/fill volume
        z_plane = a * X + b * Y + c
        dh = dem_crop - z_plane

        dh_in = dh[inside]
        dh_in = dh_in[np.isfinite(dh_in)]

        fill_vol = float(np.sum(dh_in[dh_in > 0]) * pixel_area)
        cut_vol  = float(-np.sum(dh_in[dh_in < 0]) * pixel_area)  # pozitif büyüklük
        net_vol  = float(fill_vol - cut_vol)

        # 8) 3D surface area (DEM üst yüzeyi)
        #    A3D = Σ( A_pixel * sqrt(1 + (dz/dx)^2 + (dz/dy)^2 ) )
        dz_drow, dz_dcol = np.gradient(dem_crop, px_h, px_w)  # row~y, col~x
        area_factor = np.sqrt(1.0 + dz_dcol**2 + dz_drow**2)

        af_in = area_factor[inside]
        af_in = af_in[np.isfinite(af_in)]
        surface_area_3d = float(np.sum(af_in) * pixel_area)

        results.append({
            "polygon_id": i,
            "a": a, "b": b, "c": c,
            "n_boundary_samples": int(zb.size),
            "pixel_area_m2": pixel_area,
            "fill_volume_m3": fill_vol,
            "cut_volume_m3": cut_vol,
            "net_volume_m3": net_vol,
            "surface_area_3d_m2": surface_area_3d
        })

df = pd.DataFrame(results)
print(df)

# İstersen kaydet:
out_xlsx = r"C:/AP/rock_seg/veri/bestfit_boundary_cutfill_surface.xlsx"

# Eğer klasör yoksa oluştur
os.makedirs(os.path.dirname(out_xlsx), exist_ok=True)

# Excel olarak kaydet
df.to_excel(out_xlsx, index=False, engine="openpyxl")

print("Excel file saved:", out_xlsx)

gdf["fill_v"]=df["fill_volume_m3"]
gdf["cut_v"]=df["cut_volume_m3"]
gdf["net_v"]=df["net_volume_m3"]
gdf["3d_area"]=df["surface_area_3d_m2"]
gdf.to_file(shapefile_path)