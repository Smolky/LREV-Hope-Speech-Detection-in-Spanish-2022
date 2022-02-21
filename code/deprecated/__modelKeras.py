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
import utils
import kerasModel

from tensorflow import keras
from keras import backend as K
from datasetResolver import DatasetResolver
from preprocessText import PreProcessText


# Parser
parser = argparse.ArgumentParser (description = 'Hyper-parameter optimization with TALOS')
parser.add_argument ('--dataset', dest = 'dataset', default = next (iter (config.datasets)), help = '|'.join (config.datasets.keys ()))
parser.add_argument ('--force', dest = 'force', default = False, help = 'If True, it forces to replace existing files')
parser.add_argument ('--minutes', dest = 'minutes', default = 60 * 24 * 7, type = int, help = 'Number of limits for the evaluation. Default is one week')
parser.add_argument ('--permutations', dest = 'permutations', default = None, type = int, help = 'Max number of permutations')
parser.add_argument ('--patience', dest = 'patience', default = 10, type = int, help = 'Patience for early stopping')
parser.add_argument ('--task', dest = 'task', default = '', help = 'Get the task')
parser.add_argument ('--test', dest = 'test', default = False, help = 'If true, test the best model with the test dataset')
parser.add_argument ('--features', dest = 'features', default = '', help = 'Features to focus on: lf|se|we|lf+se...')


# Parser
args = parser.parse_args ()


# @var max_words Tokenizer max words Set to None to not limit the size
max_words = None


# Remove previous intermediate files for these models
if os.path.exists (args.dataset + '_model.zip'):
    os.remove (args.dataset + '_model.zip')

for f in glob.glob (args.dataset + '/*.csv'):
    os.remove (f)

for f in glob.glob (args.dataset + '_model/*'):
    os.remove (f)

if os.path.exists (args.dataset):
    os.rmdir (args.dataset)

if os.path.exists (args.dataset + '_model'):
    os.rmdir (args.dataset + '_model')


# @var preprocess PreProcessText
preprocess = PreProcessText ()


# @var min_max_scaler MinMaxScaler To normalise the lf
min_max_scaler = sklearn.preprocessing.MinMaxScaler ()


# @var LassoReg To optimise the lf
LassoReg = sklearn.linear_model.LassoCV (normalize = False)


# Iterate over the datasets
for key, dataset_options in config.datasets[args.dataset].items ():
    
    # @var resolver DatasetResolver
    resolver = DatasetResolver ()
    
    
    # @var dataset_name String Get the dataset name
    dataset_name = args.dataset + "-" + key + '.csv'
    
    
    # @var dataset Retrieve our custom dataset
    dataset = resolver.get (dataset_name, dataset_options, args.force)
    
    
    # @var task_type string Determine if we are dealing with a regression or classification problem
    task_type = dataset_options['tasks'][args.task]['type'] if 'tasks' in dataset_options and args.task else dataset_options['type']
    
    
    # Get the dataset as a dataframe
    df = dataset.get (args.task)
    
    
    # @var true_labels_codes
    true_labels_codes = df['label'].astype ('category').cat.codes
    
    
    # @var unique_classes
    unique_classes = df['label'].unique ()
    
    
    # If the problem is classification, then we need to encode the label as numbers instead of using names
    if task_type == 'classification' and len (unique_classes) > 2:
        
        # Create a binarizer
        lb = sklearn.preprocessing.LabelBinarizer ()
        
        
        # Get unique classes
        lb.fit (df['label'].unique ())
        
        
        # Note that we are dealing with binary (one label) or multi-class (one-hot enconding)
        df_labels = pd.DataFrame (lb.transform (df['label']), columns = lb.classes_)
        df = pd.concat ([df, df_labels], axis = 1)
            
    
    # Encode labels as numbers
    elif task_type == 'classification':
        df['label'] = true_labels_codes
    
    
    # @var number_of_classes int
    number_of_classes = 1 if task_type == 'regression' or len (unique_classes) <= 2 else len (unique_classes)
    
    
    # @var feature_sets List
    feature_set = [feature_key for feature_key 
        in ['lf', 'se', 'be', 'we'] 
            if feature_key in args.features 
            or not args.features
    ]
    
    
    # @var subsets
    subsets = ['train', 'val'] if not args.test else ['test']
    
    
    # @var feature_set_cache Dict
    feature_set_cache = {features:os.path.join (config.directories['assets'], args.dataset, key, args.dataset + '-' + key + '--' + features + '.csv') 
        for features 
            in feature_set
    }
    
    
    # @var feature_dfs Get the dataframes with the features
    feature_dfs = {features: {'full': pd.read_csv (feature_set_cache[features], header = 0, sep = ',')}
        if features != 'we' else {'full': df}
            for features 
                in feature_set
    }
    
    
    # Normalization
    if 'lf' in feature_set and task_type == 'classification':
    
        # Step 1. Normalisation
        feature_dfs['lf']['full'] = pd.DataFrame (min_max_scaler.fit_transform (feature_dfs['lf']['full'].values), columns = feature_dfs['lf']['full'].columns)
        
        
        # Step 2. Feature selection
        LassoReg.fit (feature_dfs['lf']['full'].values, true_labels_codes)
        coef = pd.Series (LassoReg.coef_, index = feature_dfs['lf']['full'].columns)
        columns_to_drop = [column for column, value in coef.items () if value == 0]
        feature_dfs['lf']['full'] = feature_dfs['lf']['full'].drop (columns = columns_to_drop)
    
    
    # Specific stuff for word embeddings
    # @todo. Look to preprocess only the subsets
    if 'we' in feature_set:

        # @var preprocessing List
        preprocessing = [
            'expand_hashtags', 'remove_urls', 'remove_mentions', 'remove_digits', 'remove_whitespaces', 
            'remove_elongations', 'remove_emojis', 'to_lower', 'remove_quotations', 
            'remove_punctuation', 'remove_whitespaces', 'strip'
        ]
    
        # Custom per language
        # @todo Remove or move to another file
        if dataset_options['language'] == 'es':
            feature_dfs['we']['full']['tweet'] = preprocess.expand_acronyms (feature_dfs['we']['full']['tweet'], preprocess.msg_language)
            feature_dfs['we']['full']['tweet'] = preprocess.expand_acronyms (feature_dfs['we']['full']['tweet'], preprocess.acronyms)
        
        for pipe in preprocessing:
            feature_dfs['we']['full']['tweet'] = getattr (preprocess, pipe)(feature_dfs['we']['full']['tweet'])

    
    # Get indexes
    indexes = {subset:dataset.get_split (df, subset).index for subset in subsets}
    
    
    # Get splits for every subset
    for features in feature_set:
        for subset in subsets:
            feature_dfs[features][subset] = feature_dfs[features]['full'].loc[indexes[subset]]
    
    
    # @var Tokenizer Tokenizer is only defined if we handle word embedding features
    tokenizer = None
    
    
    # @var maxlen int It is only defined if we handle word embedding features
    maxlen = 0
    
    
    # Tokenize the word embeddings
    if 'we' in feature_set:
        
        # @var token_filename
        token_filename = os.path.join (config.directories['assets'], args.dataset, key, 'tokenizer-' + args.task + '.pickle')
        
        
        # @var token_maxlength_filename
        token_maxlength_filename = os.path.join (config.directories['assets'], args.dataset, key, 'tokenizer-' + args.task + '.txt')
        
        
        # If we are evaluating hyper-parameters, we need to create a new tokenizer and 
        # fit on the training dataset
        if not args.test:
        
            # @var Tokenizer
            tokenizer = keras.preprocessing.text.Tokenizer (num_words = max_words, oov_token = True)
            
            
            # Fit on training dataset
            tokenizer.fit_on_texts (feature_dfs['we']['train']['tweet'])
            
            
            # @var maxlen int Get the max-len size
            tokens = tokenizer.texts_to_sequences (feature_dfs['we'][subset]['tweet'])
            
            
            # Determine the maxlen
            maxlen = max (len (l) for l in tokens)
            
            
            # Store tokenizer for further use
            os.makedirs (os.path.dirname (token_filename), exist_ok = True)
            with open (token_filename, 'wb') as handle:
                pickle.dump (tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
            
            
            # Shore maxlen for further use
            os.makedirs (os.path.dirname (token_maxlength_filename), exist_ok = True)
            with open (token_maxlength_filename, 'w') as handle:
                print (maxlen, file = handle)
            
        
        # Load the tokenizer and the maxlen
        with open (token_filename, 'rb') as handle:
            tokenizer = pickle.load (handle)
            
        with open (token_maxlength_filename, 'r') as handle:
            maxlen = int (handle.read ())
            
        
        # Retrieve the tokens for the subsets we are dealing with
        tokens = {subset: 
            keras.preprocessing.sequence.pad_sequences (
                tokenizer.texts_to_sequences (feature_dfs['we'][subset]['tweet']) , padding = 'pre', maxlen = maxlen
            )
                for subset in subsets
        }
    
    
    # @var x Get features for the subsets
    x = {}
    for subset in subsets:
        x[subset] = {}
        for features in feature_set:
            x[subset][features] = tokens[subset] if features == 'we' else feature_dfs[features][subset]
    

    
    # Create a dataframe with all the labels. Notice that 
    labels_codes_df = pd.DataFrame (df['label'], index = df.index)
    labels_df = pd.DataFrame (df[lb.classes_ if number_of_classes >= 2 else 'label'], index = df.index)
    
    
    # @var y labels
    y = {subset: tensorflow.convert_to_tensor (labels_df.loc[indexes[subset]].values) for subset in subsets}
    
    
    # @var model_filename
    model_filename = os.path.join (config.directories['assets'], args.dataset, key, 'model-task-' + args.task + '-' + args.features + '.h5')
    
    
    # If we are on training mode, evaluate...
    if not args.test:
    
        # Get the optimizers for hyper-parameter optimisation
        optimizers = [keras.optimizers.Adam]
        if task_type == 'regression':
            optimizers.append (keras.optimizers.RMSprop)
        
        
        # Get the reduction_metric according to the domain problem
        reduction_metric = 'val_loss' if task_type == 'classification' else 'val_rmse'

        
        # Determine the features to evaluate
        # ['lf', 'be', 'we', 'se', 'lf+we', 'se+lf', 'se+we', 'lf+be']
        features_to_evaluate = [args.features] if args.features else feature_set
        
        
        # Determine the architectures_to_evaluate to evaluate based on the features
        architectures_to_evaluate = ['dense', 'cnn', 'lstm', 'gru', 'bilstm', 'bigru'] if 'we' in feature_set else ['dense']
        
        
        # Get parameters to evaluate
        params_epochs = [1000]
        params_lr = (0.5, 2, 10)
        params_number_of_layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        params_first_neuron = [8, 16, 48, 64, 128, 256, 512, 1024]
        params_shape = ['funnel', 'rhombus', 'long_funnel', 'brick', 'diamond', 'triangle']
        params_batch_size = [8, 16, 32, 64]
        params_dropout = [False, 0.2, 0.5, 0.8]
        params_kernel_size = [3, 5, 7]
        params_we_architecture = architectures_to_evaluate
        params_activation = ['relu', 'sigmoid', 'tanh', 'selu', 'elu']
        params_pretrained_embeddings = ['none', 'fasttext', 'glove', 'word2vec']
        params_features = features_to_evaluate
        
        
        # @var parameters_to_evaluate Parameter space Dict
        parameters_to_evaluate = {
            'task_type': [task_type],
            'tokenizer': [tokenizer],
            'name': [key],
            'dataset': [args.dataset],
            'number_of_classes': [number_of_classes],
            'epochs': params_epochs,
            'lr': params_lr,
            'optimizer': optimizers,
            'trainable': [True],
            'number_of_layers': params_number_of_layers,
            'first_neuron': params_first_neuron,
            'shape': params_shape,
            'batch_size': params_batch_size,
            'dropout': params_dropout,
            'kernel_size': params_kernel_size,
            'maxlen': [int (maxlen)],
            'we_architecture': params_we_architecture,
            'activation': params_activation,
            'pretrained_embeddings': params_pretrained_embeddings,
            'features': params_features,
            'patience': [args.patience]
        }
        
        
        # @var time_limit Used with TALOS to prevent infinite trainings times
        time_limit = (datetime.datetime.now () + datetime.timedelta (minutes = args.minutes)).strftime ("%Y-%m-%d %H:%M")
        
        
        # @var scan_object Running the experiment
        scan_object = talos.Scan (
            x = x['train'],
            x_val = x['val'],
            y = y['train'],
            y_val = y['val'],
            params = parameters_to_evaluate,
            model = kerasModel.create,
            experiment_name = args.dataset,
            time_limit = time_limit,
            minimize_loss = True,
            print_params = True,
            save_weights = True,
            round_limit = args.permutations,
            seed = config.seed,
            
            reduction_method = 'trees',
            reduction_interval = 50,
            reduction_window = 25,
            reduction_threshold = 0.2,
            reduction_metric = reduction_metric,
        )
        
        
        # Perform one hot encoding for categorical data for better understanding in a dataframe or 
        # csv file
        for feature in ['shape', 'we_architecture', 'activation', 'pretrained_embeddings']:
            if feature in scan_object.data.columns:
                scan_object.data = utils.pd_onehot (scan_object.data, feature)
        
        
        # @var params_filename String To store the hyperparameter evaluation
        params_filename = os.path.join (config.directories['assets'], args.dataset, key, 'hyperparameters-task-' + args.task + '-' + args.features + '.csv')
        
        
        # Store hyper-parameter tunning for further analysis
        os.makedirs (os.path.dirname (params_filename), exist_ok = True)
        scan_object.data.to_csv (params_filename, index = False)
        
        
        # Store the hyper-parameters in disk
        model_filename = os.path.join (config.directories['assets'], args.dataset, key, 'model-task-' + args.task + '-' + args.features + '.h5')
        os.makedirs (os.path.dirname (model_filename), exist_ok = True)
        
        
        # @var reduction_metric String Get the reduction metric to set the best model
        # @todo. For non balanced data, we need to calculate F1
        reduction_metric = 'val_accuracy' if task_type == 'classification' else 'val_loss'
        
        
        # @var reduction_order Boolean To determine if the objective is to minimise (loss) or maximise (accuracy)
        reduction_order = False if task_type == 'classification' else True
        
        
        # Retrieve best model
        best_model = scan_object.best_model (reduction_metric, asc = reduction_order)
        
        
        # Store into disk
        best_model.save (model_filename)
    
    # Predict
    if args.test:
        
        # Retrieve the best model
        best_model = keras.models.load_model (model_filename, compile = False)
        
        
        # Plot best model summary
        print (best_model.summary ())
        
        
        # In order to add the data in the same order, we need to 
        # retrieve the layer names
        layer_names = ['input_' + feature for feature in feature_set]
        
        
        # @var x_predict Dict
        x_predict = []
        for layer in best_model.layers:
            if layer.name in layer_names:
                x_predict.append (x['test'][layer_names[layer_names.index (layer.name)].replace ('input_', '')])
        
        
        # Predict (@todo multiclass)
        predictions = best_model.predict (x_predict)
        
        
        # Transform predictions to the domain problem
        if task_type == 'regression':
            print ("@todo")
            sys.exit ()
        
        else:
        
            # Binary
            if number_of_classes == 1:
                predictions = predictions >= 0.5
                predictions = np.squeeze (predictions)
            
            # Multiclass
            else:
                predictions = np.argmax (predictions, axis = 1)


        # @var true_labels Nomimal labels Get true labels as string
        y_true = labels_codes_df.loc[indexes['test']]['label'].astype ('category')
        
        
        # @var true_labels_iterable
        true_labels_iterable = dict (enumerate (y_true.cat.categories))
        
        
        # @var y_pred
        y_pred = [true_labels_iterable[int (prediction)] for prediction in predictions]
        
        
        # Classification report
        print (sklearn.metrics.classification_report (y_true, y_pred))
        
        
        # Confusion matrix
        cm = sklearn.metrics.confusion_matrix (y_true, y_pred, labels = unique_classes, normalize = 'true')
        
        utils.print_cm (cm, unique_classes)
