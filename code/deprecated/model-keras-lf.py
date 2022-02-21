"""
    Keras and Talos por hyper-parameter tunning
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

# Load basic stuff
import random
import os
import glob
import tensorflow
import talos
import datetime
import sys
import csv
import pickle
import argparse
import config
import time
import sklearn
import numpy as np
import pandas as pd
import pkbar
import utils

from tensorflow import keras
from keras import backend as K
from datasetResolver import DatasetResolver


# Parser
parser = argparse.ArgumentParser (description = 'Hyper-parameter optimization with TALOS')
parser.add_argument ('--dataset', dest = 'dataset', default = next (iter (config.datasets)), help="|".join (config.datasets.keys ()))
parser.add_argument ('--force', dest = 'force', default = False, help="If True, it forces to replace existing files")
parser.add_argument ('--minutes', dest = 'minutes', default = 60 * 24 * 7, type = int, help = "Number of limits for the evaluation. Default is one week")
parser.add_argument ('--permutations', dest = 'permutations', default = None, type = int, help = "Max number of permutations")
parser.add_argument ('--patience', dest = 'patience', default = 10, type = int, help = "Patience for early stopping")



# Parser
args = parser.parse_args ()


# @var max_words Tokenizer max words
max_words = None


# Remove previous CSVs and data
if os.path.exists (args.dataset + "_model.zip"):
    os.remove (args.dataset + "_model.zip")
    
files = glob.glob (args.dataset + '/*.csv')
for f in files:
    os.remove(f)
    
files = glob.glob (args.dataset + '_model/*')
for f in files:
    os.remove(f)


if os.path.exists (args.dataset):
    os.rmdir (args.dataset)

if os.path.exists (args.dataset + '_model'):
    os.rmdir (args.dataset + '_model')

    

"""
   create_model
   
   * @param x_train
   * @param y_train
   * @param embedding_dim
   * @param embedding_matrix
   * @param params
"""
def create_model (x_train, y_train, x_val, y_val, params):
    
    # Extract variables from params for better readability
    number_of_classes = params['number_of_classes']
    first_neuron = params['first_neuron']
    number_of_layers = params['number_of_layers']
    shape = params['shape']
    architecture = params['we_architecture']
    dropout_range = params['dropout']
    optimizer = params['optimizer']
    lr = params['lr']
    batch_size = params['batch_size']
    epochs = params['epochs']
    is_binary = number_of_classes == 1

    
    # @var last_activation_layer String Get the last activation layer based on the number of classes
    # @link https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax
    last_activation_layer = 'sigmoid' if is_binary else 'softmax'
    
    
    # @var loss_function String
    # @link https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
    loss_function = 'binary_crossentropy' if is_binary else 'categorical_crossentropy'
    
    
    # @var metric Select the metric according to the problem
    metric = keras.metrics.BinaryAccuracy (name = "accuracy") if is_binary else keras.metrics.CategoricalAccuracy (name = 'accuracy')
    
    
    # @var neurons_per_layer List Contains a list of the neurons per layer.
    neurons_per_layer = utils.get_neurons_per_layer (shape, number_of_layers, first_neuron)


    # Define the input layers
    # @var lf_input Main lf layer
    lf_input = keras.layers.Input (shape = (x_train.shape[1],))
    
    
    # Generate word embedding architecture
    # Some notes about the layers
    
    # GlobalMaxPool1D
    # @link https://stats.stackexchange.com/questions/257321/what-is-global-max-pooling-layer-and-what-is-its-advantage-over-maxpooling-layer
    
    
    # Concatenate with the deep-learning network
    x = lf_input
    for i in range (number_of_layers):
        x = keras.layers.Dense (neurons_per_layer[i])(x)
        if (dropout_range):
            x = keras.layers.Dropout (dropout_range)(x)
    layer_lf = x
    
    
    # Inputs
    inputs = [lf_input]
    
    
    # Outputs
    outputs = keras.layers.Dense (number_of_classes, activation = last_activation_layer)(layer_lf)
    
    
    # Create model
    model = keras.models.Model (inputs = inputs, outputs = outputs, name = params['name'])
    
    
    # @var Optimizer
    optimizer = optimizer (lr = talos.utils.lr_normalizer (lr, optimizer))
    

    # Compile model
    model.compile (optimizer = optimizer, loss = loss_function, metrics = [metric])

    
    # @var early_stopping Early Stopping callback
    early_stopping = tensorflow.keras.callbacks.EarlyStopping (
        monitor = 'val_loss', 
        patience = args.patience,
        restore_best_weights = True
    )
    
    
    # Fit model
    history = model.fit (
        x = x_train, 
        y = y_train,
        validation_data = (x_val, y_val),
        batch_size = batch_size,
        epochs = epochs,
        callbacks = [early_stopping]
    )
    
    
    """
    for prediction in model.predict (x_val):
        print (prediction.shape)
    """


    # finally we have to make sure that history object and model are returned
    return history, model



# @var umucorpus_ids int|string The Corpus IDs
for key, dataset_options in config.datasets[args.dataset].items ():

    print ("Processing " + key)
    
    
    # @var model_weights_filename String 
    model_weights_filename = '../results/models/{{ name }}.h5';
    
    
    # @var model_config_filename String 
    model_config_filename = '../results/models/{{ name }}.json';
    
    
    # Resolver
    resolver = DatasetResolver ()


    # Get the dataset name
    dataset_name = args.dataset + "-" + key + '.csv'

    
    # Get linguistic features
    df = pd.read_csv (os.path.join (config.directories['lf'], dataset_name), header = 0, sep = ",")
    df = df.rename (columns = {"class": "label"})
    df = df.loc[:, (df != 0).any (axis = 0)]
    
    
    # Encode label as numbers instead of user names
    lb = sklearn.preprocessing.LabelBinarizer ()
    lb.fit (df.label.unique ())
    
    
    # Binarize. 
    # Note that we are dealing with the special class 
    # where the class is binary
    if dataset_options['number_of_classes'] >= 2:
        df_labels = pd.DataFrame (lb.transform (df["label"]), columns = lb.classes_)
        df = pd.concat ([df, df_labels], axis = 1)
    else:
        df["label"] = df["label"].astype ('category').cat.codes
    
    
    
    # Divide the training dataset into training and validation
    train_df, test_df = sklearn.model_selection.train_test_split (df, train_size = dataset_options['train_size'], random_state = config.seed, stratify = df.iloc[:,-1:])
    train_df, val_df = sklearn.model_selection.train_test_split (train_df, train_size = dataset_options['val_size'], random_state = config.seed, stratify = train_df.iloc[:,-1:])
    
    
    # Parameter space
    parameters_to_evaluate = {
        'name': [key],
        'number_of_classes': [dataset_options['number_of_classes']],
        'epochs': [1000],
        'lr': (0.5, 2, 10),
        'optimizer': [keras.optimizers.Adam],
        'number_of_layers': [1, 2, 3, 4, 5, 6, 7, 8],
        'first_neuron': [8, 16, 48, 64, 128, 256],
        'shape': ['funnel', 'rhombus', 'long_funnel', 'brick', 'diamond', 'triangle'],
        'batch_size': [16, 32, 64],
        'dropout': [False, 0.2, 0.5, 0.8],
        'we_architecture': ['dense'],
        'activation': ['relu', 'sigmoid', 'tanh', 'selu', 'elu']
    }
    
    
    # Define features and labels
    if dataset_options['number_of_classes'] >= 2:
        y = tensorflow.convert_to_tensor (train_df[lb.classes_])
        y_val = tensorflow.convert_to_tensor (val_df[lb.classes_])
    else:
        y = tensorflow.convert_to_tensor (train_df['label'])
        y_val = tensorflow.convert_to_tensor (val_df['label'])
    
    
    # @var time_limit
    time_limit = (datetime.datetime.now () + datetime.timedelta (minutes = args.minutes)).strftime ("%Y-%m-%d %H:%M")
    
    
    # and run the experiment
    scan_object = talos.Scan (
        x = train_df.loc[:, ~train_df.columns.isin (['label'])], 
        x_val = val_df.loc[:, ~val_df.columns.isin (['label'])],
        y = train_df[train_df.columns[-1]],
        y_val = val_df[val_df.columns[-1]],
        params = parameters_to_evaluate,
        model = create_model,
        experiment_name = args.dataset,
        time_limit = time_limit,
        reduction_metric = 'val_loss',
        minimize_loss = True,
        print_params = True,
        round_limit = args.permutations,
        save_weights = False,
        seed = config.seed
    )
    
    
    # Store scan results for further analysis
    # We remove some values that do not changed
    columns_to_drop = []
    for column_name, value in parameters_to_evaluate.items ():
        if len(value) == 1:
            columns_to_drop.append (column_name)
    
    scan_object.data = scan_object.data.drop (columns_to_drop, axis = 1)
    
    
    # One hot encoding
    for feature in ['shape', 'activation']:
        scan_object.data = utils.pd_onehot (scan_object.data, feature)
    
    
    # Store
    params_filename = os.path.join (config.directories['assets'], args.dataset, key, 'hyperparameters-lf.csv')
    os.makedirs (os.path.dirname (params_filename), exist_ok = True)
    scan_object.data.to_csv (params_filename, index = False)


    print (params_filename)