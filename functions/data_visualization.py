import numpy as np

import matplotlib.colors as mcolors
import matplotlib
import imageio.v2 as imageio
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from functions.tf_data import *
from functions.model import *
from tabulate import tabulate

    
def compare_arrays(img_arr, msk_arr, lgnd, msk_opacity=0.3, img_brightness_factor=1.5, season = False, period = 'summer'):
    if season:
        if period == 'summer':
            bands = [0,2,4]
        if period == 'winter':
            bands = [1,3,5]
    if not season:        
        bands = [0,1,2]    
    
    new_lgnd = generate_legend_from_array(msk_arr, lgnd)
    
    # Define custom colors and colormap
    custom_colors = []
    for value in new_lgnd.values():
        custom_colors.append(value[1])
    custom_cmap = mcolors.ListedColormap(custom_colors)
    
    # Check if img_arr is not a NumPy array, then convert
    if not isinstance(img_arr, np.ndarray):
        img_arr = img_arr.numpy()

    # Check if msk_arr is not a NumPy array, then convert
    if not isinstance(msk_arr, np.ndarray):
        msk_arr = msk_arr.numpy()

    # Apply brightness adjustment to the image
    brightened_img = img_arr * img_brightness_factor
    brightened_img[brightened_img > 255] = 255  # Ensure values are within [0, 255]

    # Create a figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 16))

    # Plot the mask image
    axs[0].imshow(msk_arr, cmap=custom_cmap)
    axs[0].set_title('Mask Image')
    axs[0].axis('off')
    
    #cbar = plt.colorbar(im, ax=axs[0])
    #cbar.set_label('Mask Value')

    # Plot the original image
    axs[1].imshow(brightened_img[:, :, bands])
    axs[1].set_title('Original Image')
    axs[1].axis('off')

    # Overlay the mask on the brightened image
    axs[2].imshow(brightened_img[:, :, bands])
    axs[2].imshow(msk_arr, alpha=msk_opacity, cmap=custom_cmap)
    axs[2].set_title('Overlay')
    axs[2].axis('off')
    
    
    # Show the figure
    plt.show()

    print(f'Mask classes: {np.unique(msk_arr)}')
    print(f'Mask shape: {msk_arr.shape}')
    print(f'Image shape: {img_arr.shape}')

    
    
def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    
    
def generate_legend_from_array(array, legend):
    unique_values = np.unique(array)
    new_legend = {key: value for key, value in legend.items() if key in unique_values}
    return new_legend


def show_mask(msk_arr, lgnd, title, fig_height = 7, fig_width=4, ret=False):
    msk_arr = msk_arr.astype(np.int16)
    unique_values = np.unique(msk_arr)
    new_lgnd = generate_legend_from_array(msk_arr, lgnd)
    
    # Define custom colors and colormap
    custom_colors = []
    for value in new_lgnd.values():
        custom_colors.append(value[1])
    custom_cmap = mcolors.ListedColormap(custom_colors)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_height, fig_width))
    ax2.set_anchor('W')

    # Plot the main image with colorbar
    im = ax1.imshow(msk_arr, cmap=custom_cmap)
    #cbar = plt.colorbar(im, ax=ax1)
    ax1.set_title(title)

    # Get the unique colors from the colormap
    colors = im.cmap(im.norm(np.unique(msk_arr)))

    # Plot the colored squares with labels
    stacked_image = np.vstack([np.ones((100, 100, 4)) * color for color in colors])
    ax2.imshow(stacked_image)
    ax2.axis('off')
    for i, color in enumerate(colors):
        key = len(lgnd.keys()) - i - 1
        label = lgnd[key][0]
        ax2.text(1.05, (i + 0.5) / len(unique_values), f'{key}: {label}', transform=ax2.transAxes,
                verticalalignment='center', fontsize=10)
    #ax2.set_title('Legend')

    plt.tight_layout()
    #plt.show()
    if ret:
        return fig
    else:
        plt.show()

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Invalid hexadecimal color code")

    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    a = 1.0  # Default alpha value

    rgba_matrix = [[r, g, b, a]]
    return rgba_matrix
        

def generate_color_dict(lgnd):
    color_dict = {}
    for key, value in lgnd.items():
        color_dict[key] = value[1]
    return color_dict



def count_unique_values(array):
    unique_values, counts = np.unique(array, return_counts=True)
    count_dict = dict(zip(unique_values, counts))
    print(count_dict)
    return count_dict


def show_mask_overlay(msk_arr, img_arr, lgnd, fig_height=4, fig_width=20, ret=False, img_brightness_factor=0, opacity=0.3, season = False, period = 'summer'):
    if season:
        if period == 'summer':
            bands = [0,2,4]
        if period == 'winter':
            bands = [1,3,5]
    else:        
        bands = [0,1,2]
    msk_arr = msk_arr.astype(np.int16)
    unique_values = np.unique(msk_arr)
    #print(unique_values)
    new_lgnd = generate_legend_from_array(msk_arr, lgnd)
    
    # Define custom colors and colormap
    custom_colors = []
    for value in new_lgnd.values():
        custom_colors.append(value[1])
    color_dict = generate_color_dict(lgnd)
    custom_cmap = create_custom_colormap(color_dict)
    
    if not isinstance(img_arr, np.ndarray):
        img_arr = img_arr.numpy()

    
    brightened_img = img_arr * img_brightness_factor
    brightened_img[brightened_img > 255] = 255
    

    # Create a gridspec with different width ratios
    gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1])

    fig = plt.figure(figsize=(fig_width, fig_height))
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    
    ax1.set_anchor('E')
    ax2.set_anchor('W')
    ax3.set_anchor('W')
    ax4.set_anchor('W')
    

    # Plot the main image with colorbar
    ax1.imshow(msk_arr, cmap=custom_cmap, vmin=0, vmax=max(unique_values))
    ax1.set_title("Mask")

    # Get the unique colors from the colormap
    new_colors = []
    for value in new_lgnd.values():
        new_colors.append(hex_to_rgba(value[1]))
    
    
    # Plot the colored squares with labels
    stacked_image = np.vstack([np.ones((10, 10, 4)) * color for color in new_colors])
    ax2.imshow(stacked_image)
    ax2.axis('off')
    
    
    for i, (value, color) in enumerate(zip(unique_values, custom_colors)):
        label = new_lgnd[value][0]
        ax2.text(1.05, (len(unique_values) - i - 0.5) / len(unique_values), f'{value}: {label}', transform=ax2.transAxes,
                verticalalignment='center', fontsize=10)


    # Plot the second array (img_arr) in ax3
    ax3.imshow(brightened_img[:,:,bands])
    ax3.set_title("Satellite")

    # Overlay msk_arr and img_arr on ax4
    ax4.imshow(brightened_img[:,:,bands])
    ax4.imshow(msk_arr, cmap=custom_cmap, vmin=0, vmax=max(unique_values), alpha=opacity)  # Overlay with transparency
    ax4.set_title("Overlay")

    plt.tight_layout(w_pad=0.5)


    if ret:
        return fig
    else:
        plt.show()
        
def create_custom_colormap(color_dict):
    cmap_colors = [color_dict.get(value, '#000000') for value in range(max(color_dict.keys()) + 1)]
    cmap = mcolors.ListedColormap(cmap_colors)
    return cmap


def show_mask_pred(msk_arr, msk_pred, img_arr, lgnd, fig_height=5.5, fig_width=24, ret=False, img_brightness_factor=0, season = False, period = 'summer'):
    if season:
        if period == 'summer':
            bands = [0,2,4]
        if period == 'winter':
            bands = [1,3,5]
    else:        
        bands = [0,1,2]    
    msk_arr = msk_arr.astype(np.int16)
    unique_values = np.unique(msk_arr)
    #print(unique_values)
    new_lgnd = generate_legend_from_array(msk_arr, lgnd)
    
    # Define custom colors and colormap
    custom_colors = []
    for value in new_lgnd.values():
        custom_colors.append(value[1])
    color_dict = generate_color_dict(lgnd)
    custom_cmap = create_custom_colormap(color_dict)
    
    if not isinstance(img_arr, np.ndarray):
        img_arr = img_arr.numpy()

    
    brightened_img = img_arr * img_brightness_factor
    brightened_img[brightened_img > 255] = 255
    

    # Create a gridspec with different width ratios
    gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1]) #width_ratios=[1, 1, 1, 1, 1] if ax3 is included

    fig = plt.figure(figsize=(fig_width, fig_height))
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    # ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[2])
    ax5 = plt.subplot(gs[3])
    
    
    ax1.set_anchor('W')
    ax2.set_anchor('E')
    # ax3.set_anchor('W')
    #ax4.set_anchor('E')
    ax5.set_anchor('W')
    

    ax1.imshow(brightened_img[:,:,bands])
    ax1.set_title("Satellite")


    ax2.imshow(msk_arr, cmap=custom_cmap, vmin=0, vmax=max(unique_values))
    ax2.set_title("Mask")


    
    ax4.imshow(msk_pred, cmap=custom_cmap, vmin=0, vmax=max(unique_values))
    ax4.set_title("Prediciton")


    # Get the unique colors from the colormap
    new_colors = []
    for value in new_lgnd.values():
        new_colors.append(hex_to_rgba(value[1]))
    
    
    # Plot the colored squares with labels
    stacked_image = np.vstack([np.ones((10, 10, 4)) * color for color in new_colors])
    ax5.imshow(stacked_image)
    ax5.axis('off')
    
    
    for i, (value, color) in enumerate(zip(unique_values, custom_colors)):
        label = new_lgnd[value][0]
        ax5.text(1.05, (len(unique_values) - i - 0.5) / len(unique_values), f'{value}: {label}', transform=ax5.transAxes,
                verticalalignment='center', fontsize=10)

    plt.tight_layout(w_pad=1)


    if ret:
        return fig
    else:
        plt.show()
        
def report(models_dict, print_loss=False, print_accuracy=False, start = 0, print_params = False, params_comp = False):
    if params_comp:
        params_comparison(models_dict)
    
    print('')
    best_avg_f1_score = 0.0
    best_model = None
    sorted_models = dict(sorted(models_dict.items(), key=lambda x: x[1]['f1_avg_score'][0], reverse=True))

    for model_id, model_info in sorted_models.items():
        labels = []
        history = model_info["history"]  
        legend = model_info["lgnd"]
        avg_f1_score = model_info["f1_avg_score"][0] * 100  
        max_val_accuracy_index = history['val_accuracy'].index(max(history['val_accuracy']))
        # Get the corresponding val_loss using the same index
        corresponding_val_loss = history['val_loss'][max_val_accuracy_index]
        val_accuracy = max(model_info["history"]['val_accuracy']) * 100 
        conf_matrix =  model_info["conf_matrix"][0]
        conf_matrix_list = conf_matrix.tolist()
        scores = model_info['scores']
        
        if avg_f1_score > best_avg_f1_score:
            best_avg_f1_score = avg_f1_score
            best_model = (model_id, model_info)
        
        print(f'Model: {model_id}')
        print('------------------------------------------')
        if print_params:
            params = model_info["params"]
            for key, value in params.items():
                print(f'{key}: {value}')
            print('------------------------------------------')
            
        print('Overall Metrics:')
        print(f'Average F1 Score (Val Dataset): {avg_f1_score:.2f}%')  
        print(f'Validation Dataset Loss: {corresponding_val_loss:.4f}')  
        print(f'Validation Dataset Accuracy: {val_accuracy:.2f}%')  
        print('------------------------------------------')
        print('Confusion Matrix & Scores (Validation Dataset):')
        for key, value in legend.items():
            if key == 0:
                continue
            label = str('Class '+ str(key))
            labels.append(label)
            
        print(tabulate(conf_matrix_list, headers=labels, showindex=labels,  tablefmt='fancy_grid'))
        
        for class_name, metrics in scores[0].items():
            precision = metrics['Precision']
            recall = metrics['Recall']
            f1_score = metrics['F1 Score']
            class_description = class_name
            print(f'Class {metrics["class_index"]}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1_score:.4f}   <---->   {class_description}')
        print('')
        
        if print_loss:
            plot_info(history, 'loss', start)
            # compare_loss(history)
        if print_accuracy:
            plot_info(history, 'accuracy', start)
            # compare_accuracy(history)
    print('------------------------------------------------------------------------------------------------------------------------------')    
    print(f'Best Average F1: {best_avg_f1_score:.2f}%')
    print(f'Model: {best_model[0]}')
    best_model_dict = {'model_id': best_model[0], 'model_info': best_model[1]}
    return best_model_dict

def report_single_model(model_dict, print_loss=False, print_accuracy=False, start=0, print_params = False):
    model_info_dict = model_dict['model_info']
    params = model_info_dict["params"]
    labels = []
    model = model_info_dict["model"]
    history = model_info_dict["history"]  
    legend = model_info_dict["lgnd"]
    avg_f1_score = model_info_dict["f1_avg_score"][0] * 100  
    max_val_accuracy_index = history['val_accuracy'].index(max(history['val_accuracy']))
    # Get the corresponding val_loss using the same index
    corresponding_val_loss = history['val_loss'][max_val_accuracy_index]
    val_accuracy = max(model_info_dict["history"]['val_accuracy']) * 100 
    conf_matrix =  model_info_dict["conf_matrix"][0]
    conf_matrix_list = conf_matrix.tolist()
    scores = model_info_dict['scores']

    
    print(f'Model: {model_dict["model_id"]}')
    print('------------------------------------------')
    if print_params:
        for key, value in params.items():
            print(f'{key}: {value}')
        print('------------------------------------------')
    print('Overall Metrics:')
    print(f'Average F1 Score (Validation Dataset): {avg_f1_score:.2f}%')  
    print(f'Validation Dataset Loss: {corresponding_val_loss:.4f}')  
    print(f'Validation Dataset Accuracy: {val_accuracy:.2f}%')  
    print('------------------------------------------')
    print('Confusion Matrix & Scores (Validation Dataset):')
    for key, value in legend.items():
        if key == 0:
            continue
        label = str('Class '+ str(key))
        labels.append(label)
        
    print(tabulate(conf_matrix_list, headers=labels, showindex=labels,  tablefmt='fancy_grid'))
    
    for class_name, metrics in scores[0].items():
        precision = metrics['Precision']
        recall = metrics['Recall']
        f1_score = metrics['F1 Score']
        class_description = class_name
        print(f'Class {metrics["class_index"]}: Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1_score:.4f}   <---->   {class_description}')
    print('')
        
    if print_loss:
        plot_info(history, 'loss', start)
        #compare_loss(history)
    if print_accuracy:
        plot_info(history, 'accuracy', start)
        #compare_accuracy(history)
        
    return model, legend, params

def params_comparison(models_dict):
    best_avg_f1_score = 0.0
    best_model = None
    models_ids = {}
    row_labels = []
    table_data = []
    first_key = next(iter(models_dict))
    i=1
    row_labels.append('Model')
    for key in models_dict[first_key]['params'].keys():
        row_labels.append(key)
    row_labels.append('f1_avg_score')

    for model_id, model_info in models_dict.items():
        avg_f1_score = model_info["f1_avg_score"][0] * 100
        params = model_info["params"]
        current_list=[]
        current_list.append(model_id)
        for value in params.values():
            if isinstance(value, float):
                current_list.append(round(value,6))
            else:
                current_list.append(value)
        current_list.append(f'{avg_f1_score:.2f}%')
        table_data.append(current_list)

        if avg_f1_score > best_avg_f1_score:
            best_avg_f1_score = avg_f1_score
            best_model = (model_id, model_info)
    
    sorted_table_data = sorted(table_data, key=lambda x: float(x[-1][:-1]), reverse=True)
    
    for model_info in sorted_table_data:
        models_ids[model_info[0]]=i
        i +=1
    table_data_transposed = list(map(list, zip(*sorted_table_data)))

    for n in range(len(table_data_transposed[0])):
        table_data_transposed[0][n] = n+1
    
        
    model_table = tabulate(table_data_transposed, showindex=row_labels, tablefmt='fancy_grid')
    print('Models IDs:')
    for key, value in models_ids.items():
        print(f'{key}: {value}')
    print('------------------------------------------')
    print(model_table)
    
    print(f'Best Average F1: {best_avg_f1_score:.2f}%')
    print(f'Model: {best_model[0]}')
    print('------------------------------------------------------------------------------------------------------------------------------')
    # best_model_dict = {'model_id': best_model[0], 'model_info': best_model[1]}
    # return best_model_dict

def visual_evaluaton(processed_image_ds_train, processed_image_ds_val, processed_image_ds_test, model, legend, params, n_imgs=0):
    train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                        processed_image_ds_val, 
                                                        processed_image_ds_test, 
                                                        params['buffer_size'], 
                                                        params['batch_size']
                                                        )
    
    for image_batch, mask_batch in val_dataset.take(n_imgs):
        pred_mask_batch = model.predict(image_batch, verbose=0)
        pred_mask = create_mask(pred_mask_batch)
        show_mask_pred(mask_batch[0], pred_mask, image_batch[0], legend, img_brightness_factor=2, fig_height=5.5)
        

def full_report_single_model(processed_image_ds_train, processed_image_ds_val, processed_image_ds_test, model_dict, print_loss=False, print_accuracy=False, n_imgs=0, start=0, print_params = False):
    st = start
    print_parameters = print_params 
    model, legend, params = report_single_model(model_dict, print_loss, print_accuracy, start = st, print_params = print_parameters)
    visual_evaluaton(processed_image_ds_train, processed_image_ds_val, processed_image_ds_test, model, legend, params, n_imgs)

    
def plot_info(history, info, start):
    # Extract the validation loss values starting from the third element
    val_loss = history[info][start:]
    
    # Create a list of indices for the x-axis (epochs)
    epochs = list(range(3, len(val_loss) + 3))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_loss, marker='o', linestyle='-')
    plt.title(f'Plot of {info}')
    plt.xlabel('Epochs')
    plt.ylabel(info)
    plt.grid(True)
    plt.show()

def plot_info(history, info, start=0):
    
    if info == 'accuracy':
        col_train = 'b'
        col_val = 'r'
        
    if info == 'loss':
        col_train = 'g'
        col_val = 'purple'
            
    # Extract the training and validation values starting from the specified start index
    train_values = history[info][start:]
    val_values = history['val_' + info][start:]
    
    # Create a list of indices for the x-axis (epochs) starting from the specified start index
    epochs = list(range(start, len(train_values) + start))
    
    # Create subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot the training data
    axes[0].plot(epochs, train_values, marker='o', linestyle='-', color=col_train, label='Training')
    axes[0].set_title(f'Plot of {info} (Training)')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel(info)
    axes[0].grid(True)
    
    # Plot the validation data
    axes[1].plot(epochs, val_values, marker='o', linestyle='-', color=col_val, label='Validation')
    axes[1].set_title(f'Plot of {info} (Validation)')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel(info)
    axes[1].grid(True)
    
    # Add a legend
    axes[0].legend()
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()
    


def show_mask_pred(msk_arr, msk_pred, img_arr, lgnd, fig_height=5.5, fig_width=24, ret=False, img_brightness_factor=0, season = False, period = 'summer'):
    if season:
        if period == 'summer':
            bands = [0,2,4]
        if period == 'winter':
            bands = [1,3,5]
    else:        
        bands = [0,1,2]    
    msk_arr = msk_arr.astype(np.int16)
    unique_values = np.unique(msk_arr)
    #print(unique_values)
    new_lgnd = generate_legend_from_array(msk_arr, lgnd)
    
    # Define custom colors and colormap
    custom_colors = []
    for value in new_lgnd.values():
        custom_colors.append(value[1])
    color_dict = generate_color_dict(lgnd)
    custom_cmap = create_custom_colormap(color_dict)
    
    if not isinstance(img_arr, np.ndarray):
        img_arr = img_arr.numpy()

    
    brightened_img = img_arr * img_brightness_factor
    brightened_img[brightened_img > 255] = 255
    

    # Create a gridspec with different width ratios
    gs = plt.GridSpec(1, 4, width_ratios=[1, 1, 1, 1]) #width_ratios=[1, 1, 1, 1, 1] if ax3 is included

    fig = plt.figure(figsize=(fig_width, fig_height))
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    # ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[2])
    ax5 = plt.subplot(gs[3])
    
    
    ax1.set_anchor('W')
    ax2.set_anchor('E')
    # ax3.set_anchor('W')
    #ax4.set_anchor('E')
    ax5.set_anchor('W')
    

    ax1.imshow(brightened_img[:,:,bands])
    ax1.set_title("Satellite")


    ax2.imshow(msk_arr, cmap=custom_cmap, vmin=0, vmax=max(unique_values))
    ax2.set_title("Mask")


    
    ax4.imshow(msk_pred, cmap=custom_cmap, vmin=0, vmax=max(unique_values))
    ax4.set_title("Prediciton")


    # Get the unique colors from the colormap
    new_colors = []
    for value in new_lgnd.values():
        new_colors.append(hex_to_rgba(value[1]))
    
    
    # Plot the colored squares with labels
    stacked_image = np.vstack([np.ones((10, 10, 4)) * color for color in new_colors])
    ax5.imshow(stacked_image)
    ax5.axis('off')
    
    
    for i, (value, color) in enumerate(zip(unique_values, custom_colors)):
        label = new_lgnd[value][0]
        ax5.text(1.05, (len(unique_values) - i - 0.5) / len(unique_values), f'{value}: {label}', transform=ax5.transAxes,
                verticalalignment='center', fontsize=10)

    plt.tight_layout(w_pad=1)


    if ret:
        return fig
    else:
        plt.show()

    
def compare_accuracy(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    if isinstance(model_history, dict):
        # Plot the training set accuracy
        axs[0].plot(model_history["accuracy"])
        axs[0].set_title('Training Set accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('accuracy')

        # Plot the validation set accuracy
        axs[1].plot(model_history["val_accuracy"])
        axs[1].set_title('Validation Set accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('accuracy')
    else:
        # Plot the training set accuracy
        axs[0].plot(model_history.history["accuracy"])
        axs[0].set_title('Training Set accuracy')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('accuracy')

        # Plot the validation set accuracy
        axs[1].plot(model_history.history["val_accuracy"])
        axs[1].set_title('Validation Set accuracy')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('accuracy')

    plt.tight_layout()
    plt.show()
    
def compare_loss(model_history):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    if isinstance(model_history, dict):
        # Plot the training set accuracy
        axs[0].plot(model_history["loss"])
        axs[0].set_title('Training Set Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')

        # Plot the validation set accuracy
        axs[1].plot(model_history["val_loss"])
        axs[1].set_title('Validation Set Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')
    else:
        # Plot the training set accuracy
        axs[0].plot(model_history.history["loss"])
        axs[0].set_title('Training Set Loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')

        # Plot the validation set accuracy
        axs[1].plot(model_history.history["val_loss"])
        axs[1].set_title('Validation Set Loss')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Loss')

    plt.tight_layout()
    plt.show()
