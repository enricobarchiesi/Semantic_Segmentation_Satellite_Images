import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
import random
from functions.tf_data import *
from functions.model import *
from functions.data_visualization import *
from sklearn.model_selection import ParameterGrid
from functions.model_saving_loading import *

def random_param_modular(a, b, key=None):
    if key == 'drop_multiplier' or key == 'weight_multiplier':
        multiplier_list = []
        for value_a, value_b in zip(a, b):
            if isinstance(value_a, float) or isinstance(value_b, float) or value_a == 0 or value_b == 0 or 1 < value_a / value_b < 10:
                multiplier_list.append(random.uniform(value_a, value_b))
            else:
                low_bound, up_bound = math.log10(value_a), math.log10(value_b)
                r = random.uniform(low_bound, up_bound)
                multiplier_list.append(10 ** r)
        return multiplier_list
            
    if key == 'n_filters':

        valid_range = range(a, b + 1, 16)  # Only multiples of 16 within the range
        return random.choice(valid_range)
    elif isinstance(a, int) and isinstance(b, int):
        return random.randint(a, b)
    else:
        if isinstance(a, float) or isinstance(b, float) or a == 0 or b == 0 or 1 < a / b < 10:
            return random.uniform(a, b)
        else:
            low_bound, up_bound = math.log10(a), math.log10(b)
            r = random.uniform(low_bound, up_bound)
            return 10 ** r
        
def single_training_modular(params, 
                    processed_image_ds_train, 
                    processed_image_ds_val, 
                    processed_image_ds_test,
                    img_height,
                    img_width,
                    num_channels,
                    num_classes,
                    compare_acc=False,
                    compare_lss=False,
                    early_stopping=False,
                    restore_best=False,
                    model_summary = False):
    model={'params': params}
    train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                processed_image_ds_val, 
                                                processed_image_ds_test, 
                                                params['buffer_size'], 
                                                params['batch_size']
                                                )
        
    # Build the U-Net model with the current parameters
    unet = unet_model_modular(n_classes = num_classes, 
                              tile_width =img_width, 
                              tile_height = img_height, 
                              num_bands=num_channels, 
                              n_blocks=params['n_blocks'],
                              n_filters=params['n_filters'],
                              w_decay=params['w_decay'], 
                              droprate=params['dropout_prob'],
                              drop_multiplier=params['drop_multiplier'],
                              weight_multiplier = params['weight_multiplier'],
                              filter_growth=params['filter_growth'])
    if model_summary:
        unet.summary()
    

    
    # Compile the model
    unet.compile(
        optimizer= tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate']), # tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    restore = False
    if restore_best:
        restore = True
    
        # Define EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',      # Monitor validation accuracy
        mode='max',                  # Maximize the monitored quantity
        patience=params['early_stop_patience'],  # Number of epochs with no improvement before stopping
        restore_best_weights=restore,   # Restore weights from the epoch with the best value of the monitored quantity
        min_delta=params['early_stop_min_delta'] #At each epoch, the callback calculates the change in the monitored metric compared to the best value recorded so far. If this change (delta) is greater than or equal to min_delta, it's considered an improvement
    )
    
    # Train the model
    if early_stopping:
        model_history = unet.fit(
            train_dataset, 
            epochs=params['epochs'],
            validation_data=val_dataset,
            callbacks=[early_stopping]   
        )
    else:
        model_history = unet.fit(
            train_dataset, 
            epochs=params['epochs'],
            validation_data=val_dataset,  
        )
    
    if compare_acc:
        compare_accuracy(model_history)
    if compare_lss:
        compare_loss(model_history)
    model['model_history']=model_history
    model['model']=unet
    # model['train_ds']=train_dataset
    # model['val_ds']=val_dataset
    # model['test_ds']=test_dataset
    model['img_height']= img_height
    model['img_width']= img_width
    model['num_channels']= num_channels
    model['id']= get_current_model_name()
    
    return model        


def hyperparameters_model_grid_modular(param_grid, 
                        processed_image_ds_train, 
                        processed_image_ds_val, 
                        processed_image_ds_test,
                        img_height,
                        img_width,
                        num_channels,
                        num_classes,
                        compare_acc=False,
                        compare_lss=False,
                        early_stopping=False,
                        restore_best=False,
                        start=0,
                        model_summary = False):
    # Iterate over all parameter combinations in the grid
    all_models = {}
    unet_models = []

    for params in ParameterGrid(param_grid):
        current_model_name = get_current_model_name()

        print('')
        print('--------------------------------------------------------------------------------------------------------------')
        print(current_model_name)
        print("Testing parameters:", params)
        
        train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                      processed_image_ds_val, 
                                                      processed_image_ds_test, 
                                                      params['buffer_size'], 
                                                      params['batch_size']
                                                      )
        
        # Build the U-Net model with the current parameters
        unet = unet_model_modular(n_classes = num_classes, 
                                tile_width =img_width, 
                                tile_height = img_height, 
                                num_bands=num_channels, 
                                n_blocks=params['n_blocks'],
                                n_filters=params['n_filters'],
                                w_decay=params['w_decay'], 
                                droprate=params['dropout_prob'],
                                drop_multiplier=params['drop_multiplier'],
                                weight_multiplier = params['weight_multiplier'],
                                filter_growth=params['filter_growth'])
        if model_summary:
            unet.summary()
        # Compile the model
        unet.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        restore = False
        if restore_best:
            restore = True
        
            # Define EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',      # Monitor validation accuracy
            mode='max',                  # Maximize the monitored quantity
            patience=params['early_stop_patience'],  # Number of epochs with no improvement before stopping
            restore_best_weights=restore,   # Restore weights from the epoch with the best value of the monitored quantity
            min_delta=params['early_stop_min_delta'] #At each epoch, the callback calculates the change in the monitored metric compared to the best value recorded so far. If this change (delta) is greater than or equal to min_delta, it's considered an improvement
        )
        
        # Train the model
        if early_stopping:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,
                callbacks=[early_stopping]   
            )
        else:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,  
            )
            
            
        all_models[current_model_name] = {
            'params': params, 
            'model_history': model_history, 
            'model': unet, 
            # 'train_ds': processed_image_ds_train, 
            # 'val_ds': processed_image_ds_val, 
            # 'test_ds': processed_image_ds_test,
            'img_height': img_height,
            'img_width': img_width,
            'num_channels': num_channels
            }
        
        unet_models.append(unet)


        if compare_acc:
            compare_accuracy(model_history)
        if compare_lss:
            compare_loss(model_history)
        
        print('')
        
    return all_models, unet_models


def hyperparameters_model_modular(num_of_trials,
                        param_ranges, 
                        processed_image_ds_train, 
                        processed_image_ds_val, 
                        processed_image_ds_test,
                        img_height,
                        img_width,
                        num_channels,
                        num_classes,
                        compare_acc=False,
                        compare_lss=False,
                        early_stopping=False,
                        restore_best=False,
                        start=0,
                        model_summary = False):

    # Initialize a dictionary to store models and their parameters
    all_models = {}
    unet_models = []

    for t in range(num_of_trials):
        current_model_name = get_current_model_name()
        print('--------------------------------------------------------------------------------------------------------------')
        print(f'Model {t+1}: {current_model_name}')
        print('')
        params = {}
        for key, value in param_ranges.items():
            params[key] = random_param_modular(value[0], value[1], key=key)

        print("Testing parameters:", params)

        train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                processed_image_ds_val, 
                                                processed_image_ds_test, 
                                                params['buffer_size'], 
                                                params['batch_size']
                                                )
        
        # Build the U-Net model with the current parameters
        unet = unet_model_modular(n_classes = num_classes, 
                                tile_width =img_width, 
                                tile_height = img_height, 
                                num_bands=num_channels, 
                                n_blocks=params['n_blocks'],
                                n_filters=params['n_filters'],
                                w_decay=params['w_decay'], 
                                droprate=params['dropout_prob'],
                                drop_multiplier=params['drop_multiplier'],
                                weight_multiplier = params['weight_multiplier'],
                                filter_growth=params['filter_growth'])
        if model_summary:
            unet.summary()
        # Compile the model
        unet.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        restore = False
        if restore_best:
            restore = True
        
            # Define EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',      # Monitor validation accuracy
            mode='max',                  # Maximize the monitored quantity
            patience=params['early_stop_patience'],  # Number of epochs with no improvement before stopping
            restore_best_weights=restore,   # Restore weights from the epoch with the best value of the monitored quantity
            min_delta=params['early_stop_min_delta'] #At each epoch, the callback calculates the change in the monitored metric compared to the best value recorded so far. If this change (delta) is greater than or equal to min_delta, it's considered an improvement
        )
        
        # Train the model
        if early_stopping:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,
                callbacks=[early_stopping]   
            )
        else:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,  
            )
        

        # Store the model history, unet model, and parameters
        all_models[current_model_name] = {
            'params': params, 
            'model_history': model_history, 
            'model': unet, 
            # 'train_ds': processed_image_ds_train, 
            # 'val_ds': processed_image_ds_val, 
            # 'test_ds': processed_image_ds_test,
            'img_height': img_height,
            'img_width': img_width,
            'num_channels': num_channels
            }
        
        unet_models.append(unet)

        if compare_acc:
            compare_accuracy(model_history)
        if compare_lss:
            compare_loss(model_history)
        print('')
    
    return all_models, unet_models



       
        
def get_best_mdl_accuracy_params(all_models, unet_models):
    best_accuracy = 0.0
    best_model_params = None
    best_model = None

    for params_str, data in all_models.items():
        val_accuracy = data['model_history'].history['val_accuracy'][-1]
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_params = data['params']
            model_index = list(all_models.keys()).index(params_str)
            best_model = unet_models[model_index]

    return best_model, best_model_params, best_accuracy



def print_best_accuracy_mdl(best_accuracy, best_params):
    print("Best Validation Accuracy: {0:.2%}".format(best_accuracy))
    print('')
    print('---------------------------------')
    print("Best Model Parameters:")
    print('---------------------------------')
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"{key}: {value:.5f}")
        else:
            print(f"{key}: {value}")
    print('---------------------------------')
    
    
def model_metrics(ds, model, lgnd_dict):
    
    true_labels = []
    predicted_labels = []

    # Iterate through the dataset batches
    for image_batch, mask_batch in ds:
        pred_mask_batch = model.predict(image_batch, verbose = 0)  
        # Iterate through each item in the batch
        
        for mask in mask_batch:
            true_labels.append(tf.cast(mask, tf.int16))            
            
        for pred_mask in pred_mask_batch:
            argmax_pred_mask = create_mask_single(pred_mask)
            predicted_labels.append(argmax_pred_mask)


    # print(len(true_labels))
    # print(len(predicted_labels))
    # print(true_labels[0].shape)
    # print(predicted_labels[0].shape)
    
    true_labels_flattened = []
    predicted_labels_flattened = []

    for mask in true_labels:
        true_labels_flattened.extend(tf.reshape(mask, [-1]).numpy())
        
    for pred_mask in predicted_labels:
        predicted_labels_flattened.extend(tf.reshape(pred_mask, [-1]).numpy())    
        
    # Convert to numpy arrays
    true_labels = np.array(true_labels_flattened)
    predicted_labels = np.array(predicted_labels_flattened)

    print(f'True lables unique: {np.unique(true_labels)}')
    print(f'Predicted lables unique: {np.unique(predicted_labels)}')
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print('')
    print('----------------------------------------------')
    print('Confusion Matrix')
    print(conf_matrix)
    
    print('')
    print('----------------------------------------------')
    print('')
    
    
    average_f1, scores = prec_rec_f1(conf_matrix, lgnd_dict)
    
    return conf_matrix, average_f1, scores


def prec_rec_f1(conf_matrix, lgnd_dict):
    # Calculate precision, recall, and F1 score for each class
    precision = np.zeros(conf_matrix.shape[0])
    recall = np.zeros(conf_matrix.shape[0])
    f1_score = np.zeros(conf_matrix.shape[0])

    for class_idx in range(conf_matrix.shape[0]):
        if np.sum(conf_matrix[:, class_idx]) > 0:
            precision[class_idx] = conf_matrix[class_idx, class_idx] / np.sum(conf_matrix[:, class_idx])
        if np.sum(conf_matrix[class_idx, :]) > 0:
            recall[class_idx] = conf_matrix[class_idx, class_idx] / np.sum(conf_matrix[class_idx, :])
        if precision[class_idx] + recall[class_idx] > 0:
            f1_score[class_idx] = 2 * (precision[class_idx] * recall[class_idx]) / (precision[class_idx] + recall[class_idx])

    # Replace NaN values with 0 in precision, recall, and F1 score arrays
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)

    # Calculate average F1 score
    average_f1 = np.mean(f1_score)
    
    scores = {}

    for class_idx, (prec, rec, f1) in enumerate(zip(precision, recall, f1_score)):
        class_name = class_idx + 1
        print(f"Class {class_name}: Precision = {prec:.4f}, Recall = {rec:.4f}, F1 Score = {f1:.4f}   <---->   {lgnd_dict[class_name][0]}")
        
        class_scores = {
            'class_index': class_idx + 1,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        }

        scores[lgnd_dict[class_name][0]] = class_scores

    print('')
    print(f"Average F1 Score: {average_f1:.4f}")
    print('-----------------------------------------------------------------------------------------------------------------')
    print('')
    return average_f1, scores


def add_info_single_model(processed_image_ds_train,
                          processed_image_ds_val,
                          processed_image_ds_test,
                          single_model,
                          lgnd):
    train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
            processed_image_ds_val, 
            processed_image_ds_test,
            single_model['params']['buffer_size'],
            single_model['params']['batch_size']
            )
    conf_matrix, average_f1, scores = model_evaluation(model=single_model['model'], ds=val_dataset, lgnd_dict=lgnd, n_imgs=0)

    results = {
                'conf_matrix': [],  
                'f1_avg_score': [],  
                'scores': []
            }
    results['conf_matrix'].append(conf_matrix)
    results['f1_avg_score'].append(average_f1)
    results['scores'].append(scores)
    for key, value in results.items():
        single_model[key]=value
    return single_model

def model_evaluation(ds, model, lgnd_dict, n_imgs=0, all_models=None):
    ds_loss, ds_accuracy = model.evaluate(ds, verbose=0)
    print("Dataset loss:", f"{ds_loss:.4f}")
    print("Dataset accuracy:", f"{ds_accuracy * 100:.1f}%")
    # the bar below shows how many batches the evaluation was performed on
    
    conf_matrix, average_f1, scores = model_metrics(ds, model, lgnd_dict)
    
    for image_batch, mask_batch in ds.take(n_imgs):
        pred_mask_batch = model.predict(image_batch, verbose=0)
        pred_mask = create_mask(pred_mask_batch)
        show_mask_pred(mask_batch[0], pred_mask, image_batch[0], lgnd_dict, img_brightness_factor=2, fig_height=5.5)
    
    
    if all_models is not None:
        model_params = get_model_params(all_models, model)
        return conf_matrix, average_f1, scores, model_params
    else:
        return conf_matrix, average_f1, scores


def get_model_params(all_models, model):
    for key, data in all_models.items():
        if data['model'] is model:
            return data['params']
    return None  # Model not found in all_models
        
        
def get_specific_parameters(all_models, parameter_keys_lst): #parameter_keys has to be a list with the parameters to retrieve passed as keys 
    parameter_values_dict = {key: [] for key in parameter_keys_lst}

    for data in all_models.values():
        params = data['params']
        for parameter_key in parameter_keys_lst:
            if parameter_key in params:
                parameter_value = params[parameter_key]
                parameter_values_dict[parameter_key].append(parameter_value)

    return parameter_values_dict

def organize_datasets_with_parameters(all_models, processed_image_ds_train, processed_image_ds_val, processed_image_ds_test):
    parameter_keys = ['buffer_size', 'batch_size']
    parameter_values_dict = get_specific_parameters(all_models, parameter_keys)
    dataset_key = 100
    organized_datasets = {}
    for i in range(len(parameter_values_dict['buffer_size'])):
        train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                              processed_image_ds_val, 
                                                              processed_image_ds_test,
                                                              buffer=parameter_values_dict['buffer_size'][i], 
                                                              batch=parameter_values_dict['batch_size'][i]
                                                              )

        dataset_key = dataset_key + 1 #f"{parameter_values_dict['buffer_size'][i]}_{parameter_values_dict['batch_size'][i]}"
        organized_datasets[dataset_key] = {
            'train_dataset': train_dataset,
            'val_dataset': val_dataset,
            'test_dataset': test_dataset
        }

    return organized_datasets


def add_models_info(all_models,
                    processed_image_ds_train, 
                    processed_image_ds_val,
                    processed_image_ds_test,
                    unet_models,
                    lgnd
                    ):
    organized_datasets = organize_datasets_with_parameters(all_models, processed_image_ds_train, processed_image_ds_val, processed_image_ds_test)

    models_results={}

    for i, key, model, mdl_name in zip(range(len(organized_datasets.keys())), organized_datasets.keys(), unet_models, all_models.keys()):
        print('MODEL:' ,mdl_name)
        models_results[mdl_name] = {
            'conf_matrix': [],  
            'f1_avg_score': [],  
            'scores': [],        
            'model': [],         
            'model_params': []   
        }
        conf_matrix, average_f1, scores, model_params = model_evaluation(organized_datasets[key]['val_dataset'], model, lgnd, n_imgs=0, all_models=all_models)
        models_results[mdl_name]['conf_matrix'].append(conf_matrix)
        models_results[mdl_name]['f1_avg_score'].append(average_f1)
        models_results[mdl_name]['scores'].append(scores)
        models_results[mdl_name]['model'].append(model)
        models_results[mdl_name]['model_params'].append(model_params)
        
    best_avg_f1 = 0.0
    best_avg_f1_model = None  # Initialize the variable to store the best model
    
    for key in models_results.keys():
        avg_f1 = models_results[key]['f1_avg_score'][0]
        if avg_f1 > best_avg_f1:
            best_avg_f1 = avg_f1
            best_avg_f1_model = models_results[key]

    print(f"Best Average F1 Score: {best_avg_f1:.4f}")
    
    if best_avg_f1_model is not None:
        best_model_params = best_avg_f1_model['model_params'][0]  # Extract the parameters of the best model
            
        print('-----------------------------------------------------------------------')
        best_model_name = None
        for model_key, model_data in models_results.items():
            if model_data is best_avg_f1_model:
                best_model_name = model_key
                break
        
        if best_model_name:
            print(f"Best Average F1 Score Model Parameters ({best_model_name}):")
        else:
            print("Best Average F1 Score Model Parameters (Unknown Model):")
            
        print('-----------------------------------------------------------------------')
        for key, value in best_model_params.items():
            if isinstance(value, float):
                print(f"{key}: {value:.5f}")
            else:
                print(f"{key}: {value}")

    return best_avg_f1_model, models_results  # Return the best average F1 model



def single_training(params, 
                    processed_image_ds_train, 
                    processed_image_ds_val, 
                    processed_image_ds_test,
                    img_height,
                    img_width,
                    num_channels,
                    num_classes,
                    compare_acc=False,
                    compare_lss=False,
                    early_stopping=False,
                    restore_best=False,
                    start = 0):
    model={'params': params}
    train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                processed_image_ds_val, 
                                                processed_image_ds_test, 
                                                params['buffer_size'], 
                                                params['batch_size']
                                                )
        
    # Build the U-Net model with the current parameters
    unet = unet_model(n_classes = num_classes, tile_width =img_width, tile_height = img_height, num_bands=num_channels, n_filters=params['n_filters'], w_decay=params['w_decay'], droprate=params['dropout_prob'])
    unet.summary()
    # Compile the model
    unet.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    restore = False
    if restore_best:
        restore = True
    
        # Define EarlyStopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',      # Monitor validation accuracy
        mode='max',                  # Maximize the monitored quantity
        patience=params['early_stop_patience'],  # Number of epochs with no improvement before stopping
        restore_best_weights=restore,   # Restore weights from the epoch with the best value of the monitored quantity
        min_delta=params['early_stop_min_delta'] #At each epoch, the callback calculates the change in the monitored metric compared to the best value recorded so far. If this change (delta) is greater than or equal to min_delta, it's considered an improvement
    )
    
    # Train the model
    if early_stopping:
        model_history = unet.fit(
            train_dataset, 
            epochs=params['epochs'],
            validation_data=val_dataset,
            callbacks=[early_stopping]   
        )
    else:
        model_history = unet.fit(
            train_dataset, 
            epochs=params['epochs'],
            validation_data=val_dataset,  
        )
    
    if compare_acc:
        compare_accuracy(model_history)
    if compare_lss:
        compare_loss(model_history)
    model['model_history']=model_history
    model['model']=unet
    # model['train_ds']=train_dataset
    # model['val_ds']=val_dataset
    # model['test_ds']=test_dataset
    model['img_height']= img_height
    model['img_width']= img_width
    model['num_channels']= num_channels
    model['id']= get_current_model_name()
    
    
    
    return model
        

def hyperparameters_model_grid(param_grid, 
                        processed_image_ds_train, 
                        processed_image_ds_val, 
                        processed_image_ds_test,
                        img_height,
                        img_width,
                        num_channels,
                        num_classes,
                        compare_acc=False,
                        compare_lss=False,
                        early_stopping=False,
                        restore_best=False,
                        start=0):
    # Iterate over all parameter combinations in the grid
    all_models = {}
    unet_models = []

    for params in ParameterGrid(param_grid):
        current_model_name = get_current_model_name()

        print('')
        print('--------------------------------------------------------------------------------------------------------------')
        print(current_model_name)
        print("Testing parameters:", params)
        
        train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                      processed_image_ds_val, 
                                                      processed_image_ds_test, 
                                                      params['buffer_size'], 
                                                      params['batch_size']
                                                      )
        
        # Build the U-Net model with the current parameters
        unet = unet_model(n_classes = num_classes, tile_width =img_width, tile_height = img_height, num_bands=num_channels, n_filters=params['n_filters'], w_decay=params['w_decay'], droprate=params['dropout_prob'])
        
        # Compile the model
        unet.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )
        
        restore = False
        if restore_best:
            restore = True
        
            # Define EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',      # Monitor validation accuracy
            mode='max',                  # Maximize the monitored quantity
            patience=params['early_stop_patience'],  # Number of epochs with no improvement before stopping
            restore_best_weights=restore,   # Restore weights from the epoch with the best value of the monitored quantity
            min_delta=params['early_stop_min_delta'] #At each epoch, the callback calculates the change in the monitored metric compared to the best value recorded so far. If this change (delta) is greater than or equal to min_delta, it's considered an improvement
        )
        
        # Train the model
        if early_stopping:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,
                callbacks=[early_stopping]   
            )
        else:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,  
            )
            
            
        all_models[current_model_name] = {
            'params': params, 
            'model_history': model_history, 
            'model': unet, 
            # 'train_ds': processed_image_ds_train, 
            # 'val_ds': processed_image_ds_val, 
            # 'test_ds': processed_image_ds_test,
            'img_height': img_height,
            'img_width': img_width,
            'num_channels': num_channels
            }
        
        unet_models.append(unet)


        if compare_acc:
            compare_accuracy(model_history)
        if compare_lss:
            compare_loss(model_history)
        
        print('')
        
    return all_models, unet_models
    
    
def hyperparameters_model(num_of_trials,
                        param_ranges, 
                        processed_image_ds_train, 
                        processed_image_ds_val, 
                        processed_image_ds_test,
                        img_height,
                        img_width,
                        num_channels,
                        num_classes,
                        compare_acc=False,
                        compare_lss=False,
                        early_stopping=False,
                        restore_best=False,
                        start=0):

    # Initialize a dictionary to store models and their parameters
    all_models = {}
    unet_models = []

    for t in range(num_of_trials):
        current_model_name = get_current_model_name()
        print('--------------------------------------------------------------------------------------------------------------')
        print(f'Model {t+1}: {current_model_name}')
        print('')
        params = {}
        for key, value in param_ranges.items():
            params[key] = random_param(value[0], value[1], key=key)

        print("Testing parameters:", params)

        train_dataset, val_dataset, test_dataset = organize_ds(processed_image_ds_train, 
                                                processed_image_ds_val, 
                                                processed_image_ds_test, 
                                                params['buffer_size'], 
                                                params['batch_size']
                                                )
        
        # Build the U-Net model with the current parameters
        unet = unet_model(n_classes = num_classes, tile_width =img_width, tile_height = img_height, num_bands=num_channels, n_filters=params['n_filters'], w_decay=params['w_decay'], droprate=params['dropout_prob'])

        # Compile the model
        unet.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['learning_rate']),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=['accuracy']
        )

        restore = False
        if restore_best:
            restore = True
        
            # Define EarlyStopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',      # Monitor validation accuracy
            mode='max',                  # Maximize the monitored quantity
            patience=params['early_stop_patience'],  # Number of epochs with no improvement before stopping
            restore_best_weights=restore,   # Restore weights from the epoch with the best value of the monitored quantity
            min_delta=params['early_stop_min_delta'] #At each epoch, the callback calculates the change in the monitored metric compared to the best value recorded so far. If this change (delta) is greater than or equal to min_delta, it's considered an improvement
        )
        
        # Train the model
        if early_stopping:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,
                callbacks=[early_stopping]   
            )
        else:
            model_history = unet.fit(
                train_dataset, 
                epochs=params['epochs'],
                validation_data=val_dataset,  
            )
        

        # Store the model history, unet model, and parameters
        all_models[current_model_name] = {
            'params': params, 
            'model_history': model_history, 
            'model': unet, 
            # 'train_ds': processed_image_ds_train, 
            # 'val_ds': processed_image_ds_val, 
            # 'test_ds': processed_image_ds_test,
            'img_height': img_height,
            'img_width': img_width,
            'num_channels': num_channels
            }
        
        unet_models.append(unet)

        if compare_acc:
            compare_accuracy(model_history)
        if compare_lss:
            compare_loss(model_history)
        print('')
    
    return all_models, unet_models    


def random_param(a, b, key=None):    
    if key == 'n_filters':

        valid_range = range(a, b + 1, 16)  # Only multiples of 16 within the range
        return random.choice(valid_range)
    elif isinstance(a, int) and isinstance(b, int):
        return random.randint(a, b)
    else:
        if isinstance(a, float) or isinstance(b, float) or a == 0 or b == 0 or 1 < a / b < 10:
            return random.uniform(a, b)
        else:
            low_bound, up_bound = math.log10(a), math.log10(b)
            r = random.uniform(low_bound, up_bound)
            return 10 ** r