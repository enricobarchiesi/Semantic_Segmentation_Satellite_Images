import tensorflow as tf
import tensorflow as tf
import os
from osgeo import gdal
import numpy as np
import re
from tqdm import tqdm
    
def get_tile_references(filename):
    pattern = ".*_([0-9]+)_([0-9]+)\.tif$"
    a = re.search(pattern, filename)
    return a.group(1), a.group(2)

def find_zero_band_in_files(file_paths):
    for file_path in file_paths:
        try:
            dataset = gdal.Open(file_path)
            if dataset is not None:
                for band_index in range(1, dataset.RasterCount + 1):
                    band = dataset.GetRasterBand(band_index)
                    arr = band.ReadAsArray()
                    if 0 in arr:
                        print(f"File: {file_path}, Band: {band_index}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def find_common_elements(lists):
    if not lists:
        return []  # If the input list is empty, there are no common elements.

    common_elements = set(lists[0])  # Initialize with elements from the first list.

    for sublist in lists[1:]:
        common_elements.intersection_update(sublist)  # Update with common elements in the current sublist.

    return list(common_elements)

def get_no_nan_zero_tiles_refs(msk_tiles_folder_pths_list, img_tiles_folder_pths_list):
    keep_lst_org_lists = []
    for msk_tiles_folder_pth_list in msk_tiles_folder_pths_list:
        keep_lst_org_mask = get_no_zero_tiles_refs(msk_tiles_folder_pth_list)
        print('--------------------------------------------------------------------------------')
        keep_lst_org_lists.append(keep_lst_org_mask)
    for img_tiles_folder_pth_list in img_tiles_folder_pths_list:
        keep_lst_org_img = get_no_nan_tiles_refs(img_tiles_folder_pth_list)
        keep_lst_org_lists.append(keep_lst_org_img)
        print('--------------------------------------------------------------------------------')
    return keep_lst_org_lists


def get_no_nan_tiles_refs(tiles_folder_pth):
    files = []
    for file in os.listdir(tiles_folder_pth):
        file_path = os.path.join(tiles_folder_pth, file)
        files.append(file_path)

    print(f'Number of tiles in initial folder: {len(files)}')

    keep_lst_org = []
    
    total_files = len(files)
    with tqdm(total=total_files, desc="Processing Tiles") as pbar:
        for file_path in files:
            x, y = get_tile_references(file_path)
            tile_id = f'{x}_{y}'  # Fix the string concatenation
            
            arr = gdal.Open(file_path).ReadAsArray()
            
            if not np.isnan(arr).any():
                keep_lst_org.append(tile_id)
            
            pbar.update(1)  # Update the progress bar

    print(f'Number of tiles in cleaned folder: {len(keep_lst_org)}')
    
    return keep_lst_org

def get_no_zero_tiles_refs(tiles_folder_pth):
    files = []
    for file in os.listdir(tiles_folder_pth):
        file_path = os.path.join(tiles_folder_pth, file)
        files.append(file_path)

    print(f'Number of tiles in initial folder: {len(files)}')

    keep_lst_org = []
    
    total_files = len(files)
    with tqdm(total=total_files, desc="Processing Tiles") as pbar:
        for file_path in files:
            x, y = get_tile_references(file_path)
            tile_id = f'{x}_{y}'  # Fix the string concatenation
            
            arr = gdal.Open(file_path).ReadAsArray()
            
            if 0 not in np.unique(arr):
                keep_lst_org.append(tile_id)
            
            pbar.update(1)  # Update the progress bar

    print(f'Number of tiles in cleaned folder: {len(keep_lst_org)}')
    
    return keep_lst_org

def get_unique_values_and_paths_with_progress(folder_path):
    unique_values = set()  # Use a set to store unique values
    tiles_with_zero = []
    tiles_with_nan = []
    tiles_with_both = []

    total_files = sum(len(files) for _, _, files in os.walk(folder_path))
    with tqdm(total=total_files, desc="Processing Files") as pbar:
        for root, _, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                
                try:
                    dataset = gdal.Open(file_path)
                    if dataset is not None:
                        arr = dataset.ReadAsArray()
                        unique_values.update(np.unique(arr))
                        if 0 in arr and np.isnan(arr).any():
                            tiles_with_both.append(file_path)
                        elif 0 in arr:
                            tiles_with_zero.append(file_path)
                        elif np.isnan(arr).any():
                            tiles_with_nan.append(file_path)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
                
                pbar.update(1)  # Update the progress bar

    print("Unique values:", len(unique_values))
    print("Tiles with 0:", len(tiles_with_zero))
    print("Tiles with 'nan':", len(tiles_with_nan))
    print("Tiles with both 0 and 'nan':", len(tiles_with_both))
    return list(unique_values), tiles_with_zero, tiles_with_nan, tiles_with_both



def add_3D(cln_mask_tiles_pth):
    mask_3D_list = []
    source_folder = cln_mask_tiles_pth

    # Get a sorted list of filenames in the source folder
    sorted_filenames = sorted(filename for filename in os.listdir(source_folder) if filename.endswith(".tif"))

    for filename in sorted_filenames:
        #print(filename)
        tiff_path = os.path.join(source_folder, filename)
        ds = gdal.Open(tiff_path)
        array_2d = ds.ReadAsArray()
        array_3d = array_2d[:, :, np.newaxis]
        mask_3D_list.append(array_3d)
    
    if len(sorted_filenames) == len(mask_3D_list):
        print('----------------------------------------------')
        print('All tiles processed correclty')
        print(f'Length input list: {len(sorted_filenames)}')
        print(f'Length output list: {len(mask_3D_list)}')
    else:
        print('Error during tiles-processing')
        return 0
    
    print('')
    print('Third dimension added correctly')
    print(f'Initial shape of elements: {array_2d.shape}')
    print(f'Shape of elements in list: {mask_3D_list[0].shape}')
    print('')
    return mask_3D_list

def creating_cln_img_list(cln_img_tiles_pth):
    image_list=[]
    source_folder = cln_img_tiles_pth

    sorted_filenames = sorted(filename for filename in os.listdir(source_folder) if filename.endswith(".tif"))

    for filename in sorted_filenames:
        # print(filename)
        tiff_path = os.path.join(source_folder, filename)
        ds = gdal.Open(tiff_path)
        arr = ds.ReadAsArray()
        # print(arr.shape)
        reshaped_array = np.transpose(arr, (1, 2, 0))
        # print(reshaped_array.shape)
        image_list.append(reshaped_array)
    print('----------------------------------------------')
    print(f'Length output list: {len(image_list)}')
    print('')    
    print('Image tiles arrays transposed correctly')
    print(f'Initial shape of elements in input list: {arr.shape}')
    print(f'Final shape of elements in output list: {image_list[0].shape}')
    print('')
    return image_list


def count_files_in_folder(folder_path):
    try:
        # List all files in the folder
        files = os.listdir(folder_path)
        
        # Filter out directories and count the remaining files
        num_files = len([f for f in files if os.path.isfile(os.path.join(folder_path, f))])
        
        return num_files
    except FileNotFoundError:
        return 0  # Return 0 if the folder doesn't exist