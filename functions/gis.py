import os
import glob
import shutil

import numpy as np
from tensorflow.keras.utils import to_categorical
from osgeo import gdal
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# from pyproj import Proj, transform
# from PIL import Image
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# def composite_from_airflow(
#     airflow_pth: str,  # "airflow-runs",#"arn:aws:s3:::airflow-runs",#"/mnt/c/Users/35266/data/eo/sentinel_2/airflow/",
#     run_id: str,  # "manual__2022-10-10T08:36:08+00:00",#"s2_2018",
#     s2_composed_pth: str,
#     band_lst: list = None
# ):



#     os.makedirs("./tmp", exist_ok=True)
    

#     prdct_lst = glob.glob(f"./{run_id}/*.tif") #f"./{airflow_pth}/{run_id}/*.tif"
#     prdct_lst.sort()
#     print(prdct_lst)

#     if band_lst != None:
#         prdct_lst = [x for x in prdct_lst if any(s in x for s in band_lst)]
#         prdct_lst = sorted(prdct_lst, key=lambda x: band_lst.index([s for s in band_lst if s in x][0]))
#         print(prdct_lst)

    
#     vrt = gdal.BuildVRT(
#         "/vsimem/vrt.tif",
#         prdct_lst,
#         separate=True,
#     )

#     composite = gdal.Translate(s2_composed_pth, vrt, outputType=gdal.GDT_Float32)
#     vrt = None
#     composite = None
    # vrt_dct = {}
    # prdct_lst = glob.glob(f"./{run_id}/*.tif")  # f"{airflow_pth}/{run_id}/*.tif"
    # tile_id_lst = list(
    #     set(
    #         [
    #             os.path.basename(prdct)[0:6]
    #             for prdct in prdct_lst
    #             if os.path.basename(prdct).startswith("T3")
    #         ]
    #     )
    # )

    # date_lst = []

    # m20_bnd_lst = ["B05", "B06", "B07", "B11", "B12"]

    # for bnd in m20_bnd_lst:
    #     vrt_dct[bnd] = {}

    #     for date in list(
    #         set(
    #             [
    #                 os.path.basename(prdct_pth)[7:15]
    #                 for prdct_pth in prdct_lst
    #                 if os.path.basename(prdct_pth).split("_")[2] == bnd
    #             ]
    #         )
    #     ):
    #         if date not in date_lst:
    #             date_lst.append(date)
    #         file_pth = f"./tmp/{bnd}_{date}.tif"
    #         vrt_dct[bnd][date] = file_pth

    #         raster_lst = [
    #             prdct_pth
    #             for prdct_pth in prdct_lst
    #             if (
    #                 os.path.basename(prdct_pth).split("_")[2]
    #                 == bnd
    #                 # and os.path.basename(prdct_pth)[7:15] == str(date)
    #             )
    #         ]

    #         print(raster_lst)
    #         raster_lst.sort()
            # gdal.BuildVRT(
            #     file_pth,
            #     [
            #         prdct_pth
            #         for prdct_pth in prdct_lst
            #         if (
            #             os.path.basename(prdct_pth).startswith(tile_id)
            #             and os.path.basename(prdct_pth)[7:15] == str(date)
            #         )
            #     ],
            #     separate=True,
            # )

            # VRT = gdal.BuildVRT(
            #     "vsimem/virtual.vrt",
            #     raster_lst,
            #     separate=False,
            # )
            # output = os.system(
            #     f"gdal_merge.py -o {file_pth} -of gtiff {' '.join(raster_lst)}"
            # )  # subprocess.run
            # print(output)

            # g = gdal.Translate(file_pth, VRT)#, format="GTiff",
            #     #options=["COMPRESS=LZW", "TILED=YES"]) # if you want
            # g = None # Close file and flush to disk
            # VRT = None

    # for tile_id in tile_id_lst:
    #     vrt_dct[tile_id] = {}

    #     for date in list(
    #         set(
    #             [
    #                 os.path.basename(prdct_pth)[7:15]
    #                 for prdct_pth in prdct_lst
    #                 if os.path.basename(prdct_pth).startswith(tile_id)
    #             ]
    #         )
    #     ):
    #         if date not in date_lst:
    #             date_lst.append(date)
    #         file_pth = f"./tmp/{tile_id}_{date}.tif"
    #         vrt_dct[tile_id][date] = file_pth

    #         raster_lst = [
    #                 prdct_pth
    #                 for prdct_pth in prdct_lst
    #                 if (
    #                     os.path.basename(prdct_pth).startswith(tile_id)
    #                     and os.path.basename(prdct_pth)[7:15] == str(date)
    #                 )
    #             ]

    #         print(raster_lst)
    #         # gdal.BuildVRT(
    #         #     file_pth,
    #         #     [
    #         #         prdct_pth
    #         #         for prdct_pth in prdct_lst
    #         #         if (
    #         #             os.path.basename(prdct_pth).startswith(tile_id)
    #         #             and os.path.basename(prdct_pth)[7:15] == str(date)
    #         #         )
    #         #     ],
    #         #     separate=True,
    #         # )

    #         # output = os.system(f"gdal_merge.py -o {file_pth} -of gtiff {' '.join(raster_lst)}") #subprocess.run
    #         # print(output)

    #     g = gdal.Warp(file_pth, raster_lst, format="GTiff",
    #         options=["COMPRESS=LZW", "TILED=YES"]) # if you want
    #     g = None # Close file and flush to disk

    # for date in date_lst:
    #     file_pth = f"./tmp/20_{date}.tif"
    #     vrt_dct[date] = file_pth
    #     gdal.BuildVRT(
    #         file_pth,
    #         [
    #             vrt_dct[tile_id][date]
    #             for tile_id in tile_id_lst
    #             for date in date_lst
    #             if date in vrt_dct[tile_id].keys()
    #         ],
    #     )

    #     file_pth_10 = f"./tmp/10_{date}.tif"
    #     vrt_dct["10m"] = file_pth_10

    #     gdal.BuildVRT(
    #         file_pth_10,
    #         [
    #             prdct_pth
    #             for prdct_pth in prdct_lst
    #             if os.path.basename(prdct_pth).endswith(
    #                 "10m_composite.tif"
    #             )  # startswith("composite_10m")
    #         ],
    #     )

        # vrt = gdal.BuildVRT(
        #     "/vsimem/vrt.tif",
        #     [
        #         gdal.BuildVRT(f"./tmp/10_b0{i}.tif", vrt_dct["10m"], bandList=[i])
        #         for i in range(1, gdal.Open(vrt_dct["10m"]).RasterCount + 1)
        #     ]
        #     + [
        #         gdal.BuildVRT(f"/vsimem/b0{i}.tif", vrt_dct[date], bandList=[i])
        #         for i in range(1, gdal.Open(vrt_dct[date]).RasterCount + 1)
        #     ],
        #     separate=True,
        # )

        # [
        #     gdal.BuildVRT(f"./tmp/10_B0{i}.tif", vrt_dct["10m"], bandList=[i])
        #     for i in range(1, gdal.Open(vrt_dct["10m"]).RasterCount + 1)
        # ]

        # bnd_lst = glob.glob("./tmp/10_B0*.tif") + glob.glob(
        #     f"./{run_id}/vrt_rasters/*.tif"
        # )
        # # bnd_lst.sort()

        # bnd_lst = sorted(bnd_lst, key=lambda x: x[-7:])

        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(bnd_lst)
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        # vrt = gdal.BuildVRT(
        #     "/vsimem/vrt.tif",
        #     bnd_lst,
        #     separate=True,
        # )

    # gdal.Translate(s2_composed_pth, vrt, outputType=gdal.GDT_Float32)


def get_overlay(raster_1, raster_2, srs1=None, srs2=None):

    # Get bounds for overlay
    Xmin_1, Ymin_1, Xmax_1, Ymax_1 = GetImageBounds(raster_1)
    Xmin_2, Ymin_2, Xmax_2, Ymax_2 = GetImageBounds(raster_2)

    if srs1 != None:
        if srs1 != srs2:
            inProj = Proj(srs2)
            outProj = Proj(srs1)
            Xmin_2, Ymin_2 = transform(inProj, outProj, Xmin_2, Ymin_2)
            Xmax_2, Ymax_2 = transform(inProj, outProj, Xmax_2, Ymax_2)

    Xmin = max(Xmin_1, Xmin_2)
    Ymin = max(Ymin_1, Ymin_2)

    Xmax = min(Xmax_1, Xmax_2)
    Ymax = min(Ymax_1, Ymax_2)

    return [Xmin, Ymin, Xmax, Ymax]


def GetImageBounds(src):

    Xmin, xres, xskew, Ymax, yskew, yres = src.GetGeoTransform()
    Xmax = Xmin + (src.RasterXSize * xres)
    Ymin = Ymax + (src.RasterYSize * yres)

    #     # Get resolution
    #     R = abs(raster.GetGeoTransform()[1])

    #     # Get minimum and maximum x and y value of raster
    #     Xmin = raster.GetGeoTransform()[0]
    #     Xmax = raster.GetGeoTransform()[0] + (raster.RasterXSize * R)
    #     Ymin = raster.GetGeoTransform()[3] - (raster.RasterYSize * R)
    #     Ymax = raster.GetGeoTransform()[3]

    # Return result
    return Xmin, Ymin, Xmax, Ymax


# def reproject(in_pth, out_pth, srcSRS=None, dstSRS=None, res=None, tap=False):

#     kwargs = {}
#     ds = gdal.Warp(
#         destNameOrDestDS = in_pth,
#         srcDSOrSrcDSTab = out_pth
#         srcSRS=srcSR,
#         dstSRS=dstSRS,
#         xRes=None,
#         yRes=None,
#         targetAlignedPixels = False,
#         # resampleAlg = None

#     )

#     return ds

def map_array_to_legend(input_array, legend):
    output_array = np.zeros(input_array.shape, dtype=input_array.dtype)
    
    for old_value, new_value in legend.items():
        output_array[input_array == old_value] = new_value
    
    return output_array.astype(np.int16)

def create_cat_from_pth(in_pth, lgnd):
    ds = gdal.Open(in_pth)
    arr = ds.ReadAsArray()

    arr = arr.astype(np.int16)
    arr_out = map_array_to_legend(arr, lgnd)

    return arr_out

def create_cat(arr, lgnd):
    # map the legend from keys to values and everything else is zero
    
    logger.debug(f"Mapping old to new categories for mask")
    # logger.debug(f"Old categories count for mask: {np.unique(arr, return_counts=True)}")

    arr = arr.astype(np.int16)
    arr_out = np.zeros(arr.shape, dtype=np.int16)
    logger.debug("Created an empty array")
    for old_value, new_value in lgnd.items():
        arr_out[arr == old_value] = new_value
    logger.debug("finished mapping old to new categories")
    arr_out[~np.isin(arr, list(lgnd.keys())+[0])] = max(lgnd.values()) + 1
    # logger.debug(f"New categories count for mask: {np.unique(arr_out, return_counts=True)}")
    logger.debug("Finished mapping old to new categories")
    return arr_out


def read_mask(mask_pth, lgnd, n_classes=None):
    logger.info("Starting to process mask")
    if n_classes is None:
        n_classes = len(set(lgnd.values()))+2

    ds = gdal.Open(mask_pth)
    arr = ds.ReadAsArray()
    print(np.unique(arr))
    logger.debug(f"Original size mask: {arr.shape}")

    arr = create_cat(arr, lgnd)
    print(np.unique(arr))
    print(np.max(arr))
    
    logger.debug("Start one hot encoding")
    logger.debug(f"Shape before one hot encoding: {arr.shape}")
    
    shape = arr.shape
    os.makedirs("./tmp__", exist_ok=True)
    file_nm = "./tmp__/original.npy"
    np.save(file_nm, arr) #saving the mask with new values (coming from the lgnd) as original.npy
    
    arr_hot = to_categorical(arr, num_classes=n_classes, dtype="int16")
    logger.debug(f"Shape after one hot encoding: {arr_hot.shape}")
    logger.info("Finished processing mask")
    return arr_hot
    
    
    """
    del(arr)
    
    n_blocks_ = 9
    os.makedirs("./tmp_", exist_ok=True)
    
    n_rows_in_block = shape[0]//n_blocks_
    
    
    for i in range(0, n_blocks_-1):
        logger.debug(f"block {i+1}/{n_blocks_} processing ")
        arr_original = np.load(file_nm)
        arr_block = arr_original[i*n_rows_in_block:(i+1)*n_rows_in_block, :]
        del(arr_original)
        arr_block_hot = to_categorical(arr_block, num_classes=n_classes, dtype="int16")#int(np.max(arr))+1) #
        np.save(f"./tmp_/block_hot_{i+1}" , arr_block_hot)
        del(arr_block)
        del(arr_block_hot)
    
    logger.debug(f"row {n_blocks_}/{n_blocks_} processing ")
    arr_original = np.load(file_nm)
    arr_block = arr_original[(n_blocks_-1)*n_rows_in_block:, :]
    arr_block_hot = to_categorical(arr_block, num_classes=n_classes, dtype="int8")#int(np.max(arr))+1) #
    os.makedirs("./tmp_", exist_ok=True)
    
    np.save(f"./tmp_/block_hot_{n_blocks_}" , arr_block_hot)

    
    del(arr_original)
    
    
    arr_block_pth_lst = glob.glob("./tmp_/*.npy")
    print(arr_block_pth_lst)
    arr_block_pth_lst.sort()
    print(arr_block_pth_lst)
    
    
    arr = np.hstack([np.load(arr_pth) for arr_pth in arr_block_pth_lst[:-1]])#, axis=0)
    arr = np.concatenate((arr, np.load(arr_block_pth_lst[-1])))
    
    # arr = to_categorical(arr, num_classes=n_classes, dtype="int8")#int(np.max(arr))+1) #
    # arr = np.expand_dims(arr, -1)
    # arr = arr.astype(np.int8)  # )
    
    logger.debug(f"Shape after one hot encoding: {arr.shape}")
    logger.info("Finished processing mask")

    return arr
    """
def upload_tif(out_file_pth, arr, old_file_pth, EPSG_code):
    # out_file_pth, arr, old_file_pth, EPSG_code
    # Define the output GeoTIFF file path
    output_tif_path = out_file_pth

    # Get the dimensions of the array
    height, width = arr.shape

    # Create a new GeoTIFF file
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(output_tif_path, width, height, 1, gdal.GDT_Int16)

    # Write the array data to the GeoTIFF band
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(arr)

    # Set the geotransform (optional)
    geotransform = gdal.Open(old_file_pth).GetGeoTransform()  # Replace with your geotransform parameters
    out_ds.SetGeoTransform(geotransform)

    # Set the projection (optional)
    srs = gdal.osr.SpatialReference()
    srs.ImportFromEPSG(EPSG_code)  # Replace with your desired EPSG code
    out_ds.SetProjection(srs.ExportToWkt())

    # Close the dataset to flush data to the file
    out_ds = None

    print("GeoTIFF file created successfully.")
    
    
    
def upload_cln_msk_tiles(out_folder_pth, in_folder_pth, ref_list): 

    os.makedirs(out_folder_pth, exist_ok=True)

    # Paths to the original mask_tiles and image_tiles folders
    mask_tiles_folder = in_folder_pth
    
    # Paths to the new mask_tiles and image_tiles folders (create these folders first)
    new_mask_tiles_folder = out_folder_pth
    
    # Iterate over files in mask_tiles folder
    for filename in os.listdir(mask_tiles_folder):
        if filename.endswith(".tif"):
            parts = filename.split('_')
            if len(parts) == 5:
                reference = f"{parts[3]}_{parts[4].split('.')[0]}"
                if reference in ref_list:
                    source_path = os.path.join(mask_tiles_folder, filename)
                    dest_path = os.path.join(new_mask_tiles_folder, filename)
                    shutil.copyfile(source_path, dest_path)
            
            
                    
def upload_cln_img_tiles(out_folder_pth, in_folder_pth, ref_list): 

    os.makedirs(out_folder_pth, exist_ok=True)

    # Paths to the original mask_tiles and image_tiles folders
    image_tiles_folder = in_folder_pth

    # Paths to the new mask_tiles and image_tiles folders (create these folders first)
    new_image_tiles_folder = out_folder_pth

    # Iterate over files in image_tiles folder
    for filename in os.listdir(image_tiles_folder):
        if filename.endswith(".tif"):
            parts = filename.split('_')
            if len(parts) == 6:
                reference = f"{parts[4]}_{parts[5].split('.')[0]}"
                if reference in ref_list:
                    source_path = os.path.join(image_tiles_folder, filename)
                    dest_path = os.path.join(new_image_tiles_folder, filename)
                    shutil.copyfile(source_path, dest_path)



def create_tiles(out_folder_pth, in_folder_pth, width, height):
    for out_pth, in_pth in zip([out_folder_pth],[in_folder_pth]): 
        print(f"gdal_retile.py -ps {width} {height} -targetDir {out_pth} {in_pth}")
        os.makedirs(out_pth, exist_ok=True)
        os.system(f"gdal_retile.py -ps {width} {height} -targetDir {out_pth} {in_pth}")
        
        
        
def write_tif(fname, arr, ds_old):

    geo_transform = ds_old.GetGeoTransform()

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        fname,
        arr.shape[1],
        arr.shape[0],
        arr.shape[2],
        gdal.GDT_Int16,
    )  # Int16)
    # out_ds.SetProjection(lidar_ds.GetProjection())
    out_ds.SetGeoTransform(geo_transform)

    band = {}
    for band_nr in range(arr.shape[2]):
        band = out_ds.GetRasterBand(band_nr + 1)
        band.WriteArray(arr[:, :, band_nr])
        band.FlushCache()
        band.ComputeStatistics(False)

    out_ds.FlushCache()
    del out_ds
    del fname

def read_image(image_pth):
    ds = gdal.Open(image_pth)
    arr = ds.ReadAsArray()
    arr = arr.astype(np.int16)  # float32)
    arr = arr.transpose(1, 2, 0)
    return arr


# def write_jpg(fname, arr):
#     im = Image.fromarray(arr)
#     im.save(f"{fname}.jpeg")

def normalize(img):
    min = np.nanmin(img)
    max = np.nanmax(img)
    # min = img.min()
    # max = img.max()
    x = 2.0 * (img - min) / (max - min) - 1.0
    x = 2.0 * (img - min) / (max - min) - 1.0
    return x

def normalize_raster(in_pth:str, out_pth:str=None, nan_value=0):
    print("starting normalization")
    if out_pth is None:
        out_pth = f"{os.path.dirname(in_pth)}/norm_{os.path.basename(in_pth)}" 
    
    ds = gdal.Open(in_pth)
    driver_tiff = gdal.GetDriverByName('GTiff')
    output_im_band = driver_tiff.CreateCopy(out_pth, ds, strict=0)
    in_arr=ds.ReadAsArray()
    min_value = np.nanpercentile(in_arr,1) #np.nanmin(in_arr) #0 wv2 #0 # -1273 #
    max_value = np.nanpercentile(in_arr,99) #np.nanmax(in_arr) #625 wv2 #625 #3161 #
    print(min_value, max_value)
    # norm_nan_value = (nan_value - min_value) / (max_value - min_value)


    for band in range(ds.RasterCount):
        print(f"processing band: {band+1}")
        input_im_band = ds.GetRasterBand(band+1)
        # stats = input_im_band.GetStatistics(False, True)
        # min_value, max_value = stats[0], stats[1]
        # print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f" % (stats[0], stats[1]))
        
        input_im_band_ar = input_im_band.ReadAsArray()
        output_im_band_ar = (input_im_band_ar - min_value) / (max_value - min_value)
        output_im_band_ar[input_im_band_ar==nan_value]=np.NaN
        output_im_band.GetRasterBand(band+1).WriteArray(output_im_band_ar)
        # output_im_band.GetRasterBand(band+1).SetNoDataValue(norm_nan_value)
        input_im_band_ar = None
    
    output_im_band  = None
    
    
def normalize_inference(in_pth:str, out_pth:str=None, nan_value=0):
    if out_pth is None:
        out_pth = f"{os.path.dirname(in_pth)}/norm_{os.path.basename(in_pth)}" 
    
    ds = gdal.Open(in_pth)
    driver_tiff = gdal.GetDriverByName('GTiff')
    output_im_band = driver_tiff.CreateCopy(out_pth, ds, strict=0)
    in_arr=ds.ReadAsArray()
    min_value = 0.0 #np.nanpercentile(in_arr,1) #-906 #-1812.416 (averaged from training data) #-1850.2046 #np.nanmin(in_arr) #0.0 #
    max_value = 5710.742343994142# np.nanpercentile(in_arr,99) #9641.279 #np.nanmax(in_arr) #4950.37548828125 s2 summer #4950.37548828125 
    # norm_nan_value = (nan_value - min_value) / (max_value - min_value)


    for band in range(ds.RasterCount):
        input_im_band = ds.GetRasterBand(band+1)
        # stats = input_im_band.GetStatistics(False, True)
        # min_value, max_value = stats[0], stats[1]
        # print("[ STATS ] =  Minimum=%.3f, Maximum=%.3f" % (stats[0], stats[1]))
        
        input_im_band_ar = input_im_band.ReadAsArray()
        output_im_band_ar = (input_im_band_ar - min_value) / (max_value - min_value)
        output_im_band_ar[input_im_band_ar==nan_value]=np.NaN
        output_im_band.GetRasterBand(band+1).WriteArray(output_im_band_ar)
        # output_im_band.GetRasterBand(band+1).SetNoDataValue(norm_nan_value)
        input_im_band_ar = None
    
    output_im_band  = None