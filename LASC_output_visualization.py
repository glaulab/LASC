"""
This is the main script to get LASC predictions and visualize the LASC outputs.

@author: zhang


"""
#%% # applying LASC to get predicted labels. 

import os
os.chdir(r"D:\P\Pscript\fly paper")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from joblib import load
from tensorflow.keras.layers import Dense, LSTM, Input, Permute, Multiply, Lambda, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
import pickle

from matplotlib import colors
import matplotlib.patches as mpatches
import statistics
from collections import Counter

import LASC_output_visualization_helperFunctions as vFuncs
# include funcstions: plot_behavior_raster, get_state_transitions, plot_ethogram


# load model and model configurations
with open("D:\P\saved_models\config_LASC.pkl", "rb") as f:   #############
    loaded_data = pickle.load(f)
globals().update(loaded_data) # including class_weights, number of classes, etc.

def weighted_categorical_crossentropy(y_true, y_pred,class_weight=class_weights):
    weights = tf.reduce_sum(class_weight * y_true, axis=-1)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss * weights  # Scale loss by class weights

# load model
model = tf.keras.models.load_model(
    "D:\P\saved_models\LASC.h5", #############
    custom_objects={
    'weighted_categorical_crossentropy': weighted_categorical_crossentropy
}) 


TIME_STEPS = 50 ######################
STEP = 50 ######################

# load scaler
scaler = load("D:\P\saved_models\RobustScaler_50.joblib")  ###################

# load OneHotEncoder
enc = load("D:\P\saved_models\OneHotEncoder_50.joblib")    ###################

# load file
path = r'D:\Neu\project\7 fly\videos to be ana - fly project\\'
fileInfo = pd.read_excel(r"D:\Neu\project\7 fly\filesinfo.xlsx")
# filesinfo include mouse_id, genotype and full path of motion_features data

# define constants

stages = {
         'paralysis': 0, 
         'tonic seizure': 1, 
         'spasm': 2, 
         'clonic seizure': 3, 
         'recovery episode': 4,
         }
class_to_number = {s: i for i, s in enumerate(stages)}


# prepare help functions
def create_X(X, time_steps=1, step=1):
    Xs = []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values # each extract were a whole saved as one row, itself have data in column and several feature columns
        Xs.append(v)        
    return np.array(Xs)


def sliding_window_mode(seq, window_size):
    windows = []
    for i in range(len(seq) - window_size + 1):
        window = seq[i:i+window_size]
        window_counter = Counter(window)
        window_mode = window_counter.most_common(1)[0][0]
        windows.append(window_mode)
    return windows

def get_latent_features(model,input_data):
    
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    layer_outputs = activation_model.predict(input_data)
    layer_output = layer_outputs[8] # the output layer of the attention
    layer_output = layer_output.reshape(-1, layer_output.shape[-1])  # reshape to the long time series. 
    return layer_output
    

def complement_missed_labels(y_frame_annotation_smoothed,
                             smooth_window_size=5,
                             TIME_STEPS=50):
    '''
    the function is used to complement missed y labels after smoothing.
    Input y_frame_annotation_smoothed is a np array that have already multiplied TIME_STEPS
    
    '''
    # add labels that were missed by smoothing
    label_head = y_frame_annotation_smoothed[0]
    label_tail = y_frame_annotation_smoothed[-1]
    arr_1 = np.tile(label_head, int(((smooth_window_size-1)/2)*TIME_STEPS))
    arr_2 = np.tile(label_tail, int(((smooth_window_size-1)/2)*TIME_STEPS))
    arr_1 = arr_1.reshape(-1,1) # reshape the data to 2d. 
    arr_2 = arr_2.reshape(-1,1)
    arr = np.concatenate((arr_1, y_frame_annotation_smoothed, arr_2))
    y_frame_annotation_smoothed_complemented = np.squeeze(arr)
    return y_frame_annotation_smoothed_complemented


# Create a figure and axis object with 20 subplots
fig, axs = plt.subplots(20, 1, sharex=True, figsize=(10, 40))

# initialize vars
latent_features = []
pred_motif = []
pred_motif_smoothed = []
pred_motif_smoothed_complemented = []
mouse_id = [] # the length equal to pred_motif_smoothed_comlemented
genotype = []

fileInfo['genotype'] = fileInfo['genotype'].astype(str) # cuz 18817 is int type.
mouseId4use = fileInfo.loc[fileInfo['genotype'].isin(['18817x25004',
                                                       '18817',
                                                       'sdax25004',
                                                       'sdax18817',
                                                       ]),'mouse_id'].values

for animal_num in mouseId4use:
    fileName = fileInfo.loc[fileInfo['mouse_id']==animal_num,'featureCsv'].tolist() # find the feature.csv according to animal_id
    data_path = path + fileName[0]
    df_apply = pd.read_csv(data_path)
    df_apply.dropna(axis=0, how='any', inplace=True)
    genotype_ = fileInfo.loc[fileInfo['mouse_id']==animal_num,'genotype']
    genotype_ = genotype_.iloc[0]
    
    scale_columns = df_apply.columns.tolist()

    # create dataset
    X_apply= create_X(
        df_apply[scale_columns],
        TIME_STEPS, 
        STEP
        )
    
    # apply scalers
    reshaped_data = X_apply.reshape(-1, X_apply.shape[-1]) 
    scaled_data = scaler.transform(reshaped_data) # use the scaler generated by training dataset
    X_apply = scaled_data.reshape(X_apply.shape)

    # get latent features
    latent_features_ani = get_latent_features(model,X_apply)
    
    # get predicted labels
    y_apply_pred = model.predict(X_apply)
    y_apply_pred_decode = enc.inverse_transform(y_apply_pred).flatten()
    y_frame_annotation = np.repeat(y_apply_pred_decode,TIME_STEPS,axis=0)
    
    # smooth the prediction
    smooth_window_size = 5 #############################
    y_apply_pred_decode_smoothed = sliding_window_mode(y_apply_pred_decode,smooth_window_size)
    y_frame_annotation_smoothed = np.repeat(y_apply_pred_decode_smoothed,TIME_STEPS,axis=0)
    y_frame_annotation_smoothed = y_frame_annotation_smoothed.reshape(-1, 1)  # reshape to 2d
    y_smoothed_complemented = complement_missed_labels(
                                 y_frame_annotation_smoothed,
                                 smooth_window_size=smooth_window_size,
                                 TIME_STEPS=TIME_STEPS)
    
    # save to vars
    latent_features.append(latent_features_ani)
    pred_motif.append(y_frame_annotation)
    pred_motif_smoothed.append(y_frame_annotation_smoothed)
    pred_motif_smoothed_complemented.append(y_smoothed_complemented)
    mouse_id.append([animal_num] * len(y_smoothed_complemented))
    genotype.append([genotype_]*len(y_smoothed_complemented))
    
    # plot behavior rasters
    vFuncs.plot_behavior_raster(
        y_smoothed_complemented,
        animal_num,
        axs,
        0,
        len(y_smoothed_complemented)
        )
    
fig.suptitle(f"smooth window size:{smooth_window_size}")
plt.tight_layout()
plt.show()

#%% get the transition features, stage structure for each line. 
# '18817x25004 as e.g. 



animal_grp = {'18817x25004': fileInfo.loc[fileInfo['genotype'].isin(['18817x25004'
                                                       ]),'mouse_id'].values,
              '18817': fileInfo.loc[fileInfo['genotype'].isin(['18817'
                                                       ]),'mouse_id'].values,
              'sdax25004': fileInfo.loc[fileInfo['genotype'].isin(['sdax25004'
                                                       ]),'mouse_id'].values,
              'sdax18817': fileInfo.loc[fileInfo['genotype'].isin(['sdax18817'
                                                       ]),'mouse_id'].values} # group name and animal num



transitions = []
transition_matrix = []
transition_matrix_prob = []

# initiate vars
motif_usage_frameNum = np.zeros((len(animal_grp['18817x25004']),len(stages))) # dim1:animal, dim2: motif
motif_usage_pct = np.zeros((len(animal_grp['18817x25004']),len(stages))) 

n = 0
for animal_num in animal_grp['18817x25004']: 
    
    state_transitions, trans_mt, trans_mt_prob = vFuncs.get_state_transitions(pred_motif_smoothed_complemented[animal_num])
    transitions.append(state_transitions)
    transition_matrix.append(trans_mt)
    transition_matrix_prob.append(trans_mt_prob)

    unique, counts = np.unique(pred_motif_smoothed_complemented[animal_num], return_counts=True)
    ls_ = list(zip(unique,counts))
    for item in ls_:
        motif_usage_frameNum[n,stages[item[0]]] = item[1]
        motif_usage_pct[n,stages[item[0]]] = item[1] /len(pred_motif_smoothed_complemented[animal_num])
    n = n+1

#%% plot ethogram


'''
# Sample input data
nodes = ['A', 'B', 'C', 'D', 'E']
node_size = [300, 200, 150, 100, 250]
node_color = ['red', 'blue', 'green', 'yellow', 'purple']
node_XY = {'A': (0, 0), 'B': (1, 1), 'C': (2, 0), 'D': (1, -1), 'E': (3, 0)}

edges = [('A', 'B', 0.8, 'blue'), ('B', 'C', 0.6, 'red'), ('C', 'D', 0.5, 'green'), ('D', 'E', 0.7, 'purple')]
    
'''

genotype = '18817x25004'

nodes = ['P','T','S','C','R']

node_size = np.mean(motif_usage_pct,axis = 0)
node_size = node_size * 5000    # add a scale factor for plotting
node_size = node_size.tolist()

node_color = ['blue', 'red', 'orange', 'purple', 'green'] # according to PTSCR

node_XY= {'P': (2, 6), 'T': (1,2.4), 'S': (4, 0), 'C': (7, 2.4), 'R': (6, 6)}

edges = []

mapping = {
         'P': 0, 
         'T': 1, 
         'S': 2, 
         'C': 3, 
         'R': 4,
         }

colormap = {'P': 'blue',
            'T': 'red',
            'S': 'orange',
            'C': 'purple',
            'R': 'green'}

# Generate all possible combinations
all_trans = [row + column for row in nodes for column in nodes]
all_trans.remove('PP')
all_trans.remove('TT')
all_trans.remove('SS')
all_trans.remove('CC')
all_trans.remove('RR')

stacked_arrays = np.stack(transition_matrix_prob)    # Shape: (n, 5, 5)
stacked_arrays[np.isnan(stacked_arrays)] = 0    # Replace NaN with 0
transition_matrix_prob_mean = np.mean(stacked_arrays, axis=0)  # average across animals
                                                               # shape (5,5)

for trans in all_trans:
    edge = (trans[0], trans[1])
    edge = edge + (transition_matrix_prob_mean[mapping[trans[0]], mapping[trans[1]]], colormap[trans[0]])
    edges.append(edge)    


vFuncs.plot_ethogram(nodes=nodes,
                  node_size=node_size,
                  node_color=node_color,
                  node_XY=node_XY,
                  edges=edges,
                  )
plt.title(genotype)
    






