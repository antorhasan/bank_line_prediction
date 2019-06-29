import rasterio

raster = rasterio.open(fp)

print(raster.meta)