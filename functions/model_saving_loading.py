import os
import shutil
import tensorflow as tf
import imageio.v2 as imageio
from functions.tf_data import *
from functions.model import *
from functions.data_visualization import *
from sklearn.model_selection import ParameterGrid
import datetime
import pickle
from tensorflow.keras.models import load_model

def get_current_model_folder_name():
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    
    datetime_string = f"models_folder_{day:02d}_{month:02d}_{year}-{hour:02d}h_{minute:02d}m_{second:02d}s"
    return datetime_string

def get_current_model_name():
    now = datetime.datetime.now()
    year = now.year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    
    datetime_string = f"model_{day:02d}_{month:02d}_{year}-{hour:02d}h_{minute:02d}m_{second:02d}s"
    return datetime_string

def save_models_and_info(all_models, save_dir, lgnd_dict):
    
    folder_name = get_current_model_folder_name()
    FOLDER_DIR = f'{save_dir}/{folder_name}'
        
    # Create the save directory if it doesn't exist
    os.makedirs(FOLDER_DIR, exist_ok=True)
    
    
    # Save each model's history and model separately
    for model_name, model_info in all_models.items():
        model_folder = f"{FOLDER_DIR}/{model_name}"
        os.makedirs(model_folder, exist_ok=True)
        
        # Save model history
        model_history = model_info['model_history'].history
        with open(f'{model_folder}/{model_name}_history.pkl', 'wb') as f:
            pickle.dump(model_history, f)

        # Save model architecture and weights
        model = model_info['model']
        model.save(f'{model_folder}/{model_name}_model')
        
        # Save model parameters
        model_params = model_info['params']
        with open(f'{model_folder}/{model_name}_params.pkl', 'wb') as f:
            pickle.dump(model_params, f)
            
        # Save model conf_matrix
        model_conf_matrix = model_info['conf_matrix']
        with open(f'{model_folder}/{model_name}_conf_matrix.pkl', 'wb') as f:
            pickle.dump(model_conf_matrix, f)   

        # Save model scores
        model_scores = model_info['scores']
        with open(f'{model_folder}/{model_name}_scores.pkl', 'wb') as f:
            pickle.dump(model_scores, f)   

            
        with open(f'{FOLDER_DIR}/legend.pkl', 'wb') as f:
            pickle.dump(lgnd_dict, f)

        # Save model f1_avg_score
        model_f1_avg_score = model_info['f1_avg_score']
        with open(f'{model_folder}/{model_name}_f1_avg_score.pkl', 'wb') as f:
            pickle.dump(model_f1_avg_score, f)  
            
        print(f"Saved model '{model_name}' and its information in '{model_folder}'.")
        
def save_single_model_and_info(model_info, save_dir, lgnd_dict):
    
    folder_name = get_current_model_folder_name()
    FOLDER_DIR = f'{save_dir}/{folder_name}'
        
    # Create the save directory if it doesn't exist
    os.makedirs(FOLDER_DIR, exist_ok=True)
    
    model_id = model_info['id']
    model_folder = f"{FOLDER_DIR}/{model_id}"
    os.makedirs(model_folder, exist_ok=True)
    
    # Save model history
    model_history = model_info['model_history'].history
    with open(f'{model_folder}/{model_id}_history.pkl', 'wb') as f:
        pickle.dump(model_history, f)

    # Save model architecture and weights
    model = model_info['model']  # Access the model from the dictionary
    model.save(f'{model_folder}/{model_id}_model')
    
    # Save model parameters
    model_params = model_info['params']
    with open(f'{model_folder}/{model_id}_params.pkl', 'wb') as f:
        pickle.dump(model_params, f)
    
    # Save model conf_matrix
    model_conf_matrix = model_info['conf_matrix']
    with open(f'{model_folder}/{model_id}_conf_matrix.pkl', 'wb') as f:
        pickle.dump(model_conf_matrix, f)   
    
    # Save model scores
    model_scores = model_info['scores']
    with open(f'{model_folder}/{model_id}_scores.pkl', 'wb') as f:
        pickle.dump(model_scores, f)   
    
    # Save model f1_avg_score
    model_f1_avg_score = model_info['f1_avg_score']
    with open(f'{model_folder}/{model_id}_f1_avg_score.pkl', 'wb') as f:
        pickle.dump(model_f1_avg_score, f)   
    
    with open(f'{FOLDER_DIR}/legend.pkl', 'wb') as f:
        pickle.dump(lgnd_dict, f)
        
    print(f"Saved model '{model_id}' and its information in '{model_folder}'.")


def load_model_history(history_path):
    if not os.path.exists(history_path):
        print('Path error')
        return None  # Return None if the history file doesn't exist
    with open(history_path, 'rb') as f:
        model_history = pickle.load(f)
    return model_history

def load_models_from_directory(directory): #'models_weights/models_folder_02_09_2023-12h_05m_46s'
    models_data = {}
    legend_path = f'{directory}/legend.pkl'
    # Load the history
    with open(legend_path, 'rb') as f:
        legend = pickle.load(f)
    # List all subdirectories
    subdirectories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(directory, subdirectory)
        model_path = os.path.join(subdirectory_path, f"{subdirectory}_model")
        history_path = os.path.join(subdirectory_path, f"{subdirectory}_history.pkl")
        params_path = os.path.join(subdirectory_path, f"{subdirectory}_params.pkl")
        conf_matrix_path = os.path.join(subdirectory_path, f"{subdirectory}_conf_matrix.pkl")
        scores_path = os.path.join(subdirectory_path, f"{subdirectory}_scores.pkl")
        f1_avg_score_path = os.path.join(subdirectory_path, f"{subdirectory}_f1_avg_score.pkl")
        
        # Check if both model and history files exist for this subdirectory
        if os.path.exists(model_path) and os.path.exists(history_path) and os.path.exists(params_path):
            # Load the model
            model = load_model(model_path)
            
            # Load the history
            with open(history_path, 'rb') as f:
                history = pickle.load(f)
            
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            
            with open(conf_matrix_path, 'rb') as f:
                conf_matrix = pickle.load(f)
                
            with open(scores_path, 'rb') as f:
                scores = pickle.load(f)
                
            with open(f1_avg_score_path, 'rb') as f:
                f1_avg_score = pickle.load(f)
            
            # Store the model and history in a dictionary
            models_data[subdirectory] = {
                'model': model,
                'history': history,
                'params': params,
                'conf_matrix': conf_matrix,
                'scores': scores,
                'f1_avg_score': f1_avg_score,
                'lgnd': legend
            }

    return models_data

def merge_model_info(all_models, results):
    merged_info = {}  # Initialize the merged dictionary
    
    for model_id, model_data in all_models.items():
        if model_id in results:
            # Merge the information from 'all_models' and 'results' for the same model ID
            merged_info[model_id] = {
                **model_data,  # Include information from 'all_models'
                'conf_matrix': results[model_id]['conf_matrix'],
                'f1_avg_score': results[model_id]['f1_avg_score'],
                'scores': results[model_id]['scores']
            }
        else:
            # If there's no corresponding result for the model, simply copy information from 'all_models'
            merged_info[model_id] = model_data
    
    return merged_info

        
def delete_directory_or_file(path):
    if os.path.isfile(path):
        # If the path is a file, simply delete it
        try:
            os.remove(path)
            print(f"Deleted file: {path}")
        except OSError as e:
            print(f"Error deleting file: {path} - {e}")
    elif os.path.isdir(path):
        # If the path is a directory, delete it and its contents
        try:
            shutil.rmtree(path)
            print(f"Deleted directory and its contents: {path}")
        except Exception as e:
            print(f"Error deleting {path} - {e}")
    else:
        print(f"Invalid path: {path}")