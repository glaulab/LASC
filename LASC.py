"""
This script is for LASC model construction and compilation.

@author: zhang


"""

import os
os.chdir(r"D:\P\Pscript\fly paper")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, Input, Permute, Multiply, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


# LASC construction

def attention_layer(inputs):
    attention = Dense(1, activation= 'tanh')(inputs) # output dim:(none, 50, 1)
    attention = Lambda(lambda x: tf.squeeze(x, axis=-1))(attention) # (none, 50)
    attention = Lambda(lambda x: tf.keras.activations.softmax(x, axis=1))(attention) # (none, 50)
    attention = Lambda(lambda x: tf.expand_dims(x, axis=-1))(attention) # (none, 50, 1)
    return Multiply()([inputs, attention]) # (none, 50, 64)

def create_LASC_model(input_shape, num_classes):
    # input_shape represents the shape of your input data
    # num_classes represents the number of classes in your classification task
    inputs = Input(shape=input_shape)
    lstm_output = LSTM(64, return_sequences=True,
                       dropout=0.2,
                       recurrent_dropout=0.2)(inputs)
    
    attention_output_reshaped = tf.transpose(lstm_output, [0, 2, 1])
    attention_output = attention_layer(attention_output_reshaped) # on feature axis
    attention_output = tf.transpose(attention_output, [0, 2, 1]) # reshape back

    attention_output = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention_output) # with attention

    outputs = Dense(num_classes, activation='softmax')(attention_output)
    
    model = Model(inputs=inputs, outputs=outputs)
    # return model
    return model

model = create_LASC_model(input_shape=(50,159)) # (timestep, feature_num)

#%% LASC model compilation

# Compute class weights to handle imbalance
def cal_class_weight(y_train,num_classes):
    class_weights = compute_class_weight(class_weight='balanced', 
                                         classes=np.unique(y_train.argmax(axis=1)), 
                                         y=y_train.argmax(axis=1))
    class_weight_dict = {i: class_weights[i] for i in range(num_classes)}

    return class_weights,class_weight_dict


# Custom loss function to apply weights
def weighted_categorical_crossentropy(y_true, y_pred,class_weight):
    
    weights = tf.reduce_sum(class_weight * y_true, axis=-1)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss * weights  # Scale loss by class weights

model.compile(
              loss=weighted_categorical_crossentropy, 
              optimizer='adam', 
              metrics=['accuracy', tf.keras.metrics.Precision(), 
                       tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])


# Shuffle the data manually with a seed
np.random.seed(42)
indices = np.arange(len(X_train))
np.random.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]

history = model.fit(X_train, 
                    y_train, 
                    epochs=100, 
                    batch_size=16, 
                    validation_data=(X_val, y_val),
                    class_weight=class_weight_dict,
                    callbacks=EarlyStopping(patience=15, restore_best_weights=True),
                    shuffle=False,
                    )
