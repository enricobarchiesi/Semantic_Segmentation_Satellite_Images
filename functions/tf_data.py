import tensorflow as tf
from keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


def process_arrays_64(image_array, mask_array):
    img = tf.convert_to_tensor(image_array, dtype=tf.float64)

    mask = tf.convert_to_tensor(mask_array)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.cast(mask, tf.float32)
    return img, mask

def process_arrays_32(image_array, mask_array):
    img = tf.convert_to_tensor(image_array, dtype=tf.float32)

    mask = tf.convert_to_tensor(mask_array)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    mask = tf.cast(mask, tf.float32)
    return img, mask

def create_mask(pred_mask_batch): # the mask is inside a batch
    pred_mask = tf.argmax(pred_mask_batch, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0] # extracts the first element of a batch


def create_mask_single(pred_mask): 
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask

def organize_ds(train, val, test, buffer, batch):
    train_dataset = train.cache().shuffle(buffer).batch(batch)
    val_dataset = val.cache().shuffle(buffer).batch(batch)
    test_dataset = test.cache().shuffle(buffer).batch(batch)
    return train_dataset, val_dataset, test_dataset

def train_val_test_data_split(lst, train_perc, test_perc):
    # Generate indices for splitting while keeping correspondence
    indices = list(range(len(lst)))
    train_indices, temp_indices = train_test_split(indices, test_size=1-train_perc, random_state=42)  
    val_indices, test_indices = train_test_split(temp_indices, test_size=test_perc, random_state=42)  
    
    return train_indices, val_indices, test_indices

def create_train_test_lists(lst, train_indices, val_indices, test_indices):
    # Create the sublists
    list_train = [lst[i] for i in train_indices]
    list_val = [lst[i] for i in val_indices]
    list_test = [lst[i] for i in test_indices]
    
    return list_train, list_val, list_test

def dataset_split(image_lists, mask_3D_lists, train_split, test_split):
    
    train_indices, val_indices, test_indices = train_val_test_data_split(lst= image_lists[0], train_perc = train_split, test_perc = test_split)
    
    mask_list_train = [] 
    mask_list_val = [] 
    mask_list_test = []
    
    image_list_train = [] 
    image_list_val = [] 
    image_list_test = []
    
    for mask_list in mask_3D_lists:
        msk_train, msk_val, msk_test = create_train_test_lists(mask_list, train_indices, val_indices, test_indices)    
        for i in msk_train:
            mask_list_train.append(i)
        for i in msk_val:
            mask_list_val.append(i)
        for i in msk_test:
            mask_list_test.append(i)
    
    for img_list in image_lists:
        img_train, img_val, img_test = create_train_test_lists(img_list, train_indices, val_indices, test_indices)    
        for i in img_train:
            image_list_train.append(i)
        for i in img_val:
            image_list_val.append(i)
        for i in img_test:
            image_list_test.append(i)
    
    return mask_list_train, mask_list_val, mask_list_test, image_list_train, image_list_val, image_list_test

def create_ds(image_list_ds, mask_list_ds, byte_type):
    image_array_list_train = image_list_ds  # List of image arrays
    mask_array_list_train = mask_list_ds   # List of mask arrays

    # Convert the lists to numpy arrays
    image_array = np.array(image_array_list_train)
    mask_array = np.array(mask_array_list_train)

    # Create a TensorFlow dataset using the numpy arrays
    image_dataset = tf.data.Dataset.from_tensor_slices(image_array)
    mask_dataset = tf.data.Dataset.from_tensor_slices(mask_array)

    # Combine the image and mask datasets into pairs using zip
    paired_dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
    
    tensor_list_images = [tf.constant(array) for array in image_array_list_train]
    tensor_list_masks = [tf.constant(array) for array in mask_array_list_train]

    # Create a TensorFlow dataset from the tensor list
    train_dataset = tf.data.Dataset.from_tensor_slices((tensor_list_images,tensor_list_masks))
    
    if byte_type == 64:
        processed_image_ds = train_dataset.map(process_arrays_64)
        
    if byte_type == 32:
        processed_image_ds = train_dataset.map(process_arrays_32)
    
    return processed_image_ds

def bt (res):
    if res == 5:
        byte_type = 32
        return byte_type
    if res == 10:
        byte_type = 64
        return byte_type
"""
def train_val_test_split(image_list, mask_list, train_perc):
    # Assuming image_list, mask_list, and test_list contain your 3D arrays

    # Generate indices for splitting while keeping correspondence
    indices = list(range(len(image_list)))
    train_indices, temp_indices = train_test_split(indices, test_size=1-train_perc, random_state=42)  
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)  

    # Create the sublists
    image_list_train = [image_list[i] for i in train_indices]
    mask_list_train = [mask_list[i] for i in train_indices]

    image_list_val = [image_list[i] for i in val_indices]
    mask_list_val = [mask_list[i] for i in val_indices]

    image_list_test = [image_list[i] for i in test_indices]
    mask_list_test = [mask_list[i] for i in test_indices]

    return image_list_train, mask_list_train, image_list_val, mask_list_val, image_list_test, mask_list_test
"""
