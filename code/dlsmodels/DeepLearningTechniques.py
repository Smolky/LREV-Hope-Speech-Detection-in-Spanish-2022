import os
import glob
import tensorflow
import datetime
import sys
import csv
import json
import itertools
import pickle
import time
import sklearn
import numpy as np
import pandas as pd
import bootstrap
import traceback
import shutil
import multiprocessing
import tempfile


from pathlib import Path
from contextlib import redirect_stdout


import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune import CLIReporter
from ray.tune import Callback
from hyperopt import hp

from . import utils
from contextlib import redirect_stdout
from tqdm import tqdm

from sklearn import preprocessing
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from .BaseModel import BaseModel
from . import kerasModel
from features.FeatureResolver import FeatureResolver

class DeepLearningTechniques (BaseModel):
    """
    DeepLearningTechniques
    
    Keras por hyper-parameter tunning that include 
    recurrent neural networks, convolutional neural networsks, 
    and vanilla multilayer perceptrons
    """
    
    # @var all_available_features List
    all_available_features = ['lf', 'se', 'be', 'we', 'ne', 'cf', 'bf', 'pr', 'ng', 'it', 'bi', 'rf']
    
    
    # @var log Boolean
    log = False
    

    @classmethod
    def adapt_argument_parser (cls, parser):
        
        # @var choices List of list
        choices = FeatureResolver.get_feature_combinations (['lf', 'se', 'be', 'we', 'ne', 'cf', 'bf', 'pr', 'ng', 'it', 'bi', 'rf'])
        
        
        # Add parser
        parser.add_argument ('--features', 
            dest = 'features', 
            default = 'all', 
            help = 'Select the family or features to evaluate', 
            choices = ['all'] + ['-'.join (choice) for choice in choices]
        )
        
    
    
    def get_folder (self):
        """
        @inherit
        """
        return self.folder or 'deep-learning-' + '-'.join (self.get_feature_combinations ())

    
    def get_main_metric (self):
        """ 
        The metric that determines which is the best model. Note that 
        most of the metrics should be maximised. However, some metrics 
        such as loss should be minimised
        
        @return String
        """
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
        
        # @var dataset_options Dict
        dataset_options = self.dataset.get_task_options ()
        
        
        # Check in the main metric has been specifically defined in the dataset
        if 'scoring' in dataset_options:
            return dataset_options['scoring']
            
            
        # If the task is for regression...
        if 'regression' == task_type:
            return 'val_rmse'

        
        # If the task is multi_label classification...
        if 'multi_label' == task_type:
            return 'val_f1_score'
        
        
        # Classification metrics
        # Imbalanced datasets require f1 score to determine the correct balanced among classes
        # However, we include the accuracy as well in order to see if the model is learning to 
        # classify instances
        if 'classification' == task_type:
            return 'val_f1_score' if self.dataset.is_imbalanced () else 'val_accuracy'


        # Most generic option
        return 'val_loss'
    

    def load_features (self):

        # @var feature_resolver FeatureResolver
        feature_resolver = FeatureResolver (self.dataset)
        
        
        # @var task_type
        task_type = self.dataset.get_task_type ()
        
        
        # @var feature_combinations List
        feature_combinations = FeatureResolver.get_feature_combinations (self.get_available_features ()) if self.args.features == 'all' else [self.args.features.split ('-')]
        
        
        # Load all the available features
        for features in feature_combinations:
            
            # Indicate which features we are loading
            print ("loading features...")
            
            for feature_set in features:

                # @var feature_file String
                feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)

            
                # @var features_cache String The file where the features are stored
                features_cache = self.dataset.get_working_dir (self.dataset.task, feature_file)
                
                
                # If the feautures are not found, get the default one
                if not Path (features_cache).is_file ():
                    features_cache = self.dataset.get_working_dir (self.dataset.task, feature_set + '.csv')
                
                
                # Indicate what features are loaded
                print ("\t" + features_cache)
                if not Path (features_cache).is_file ():
                    print ("skip...")
                    continue
            
            
                # Set features
                self.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))

    
    
    def has_external_features (self):
        """
        @inherit
        """
        return True
        

    def train (self, force = False, using_official_test = True):
        """
        @inherit
        """
        
        # @var dataset_options Dict
        dataset_options = self.dataset.get_task_options ()
        
        
        # @var df DataFrame The Dataset
        df = self.dataset.get ()

        
        # Load the features, to speed-up the process
        self.load_features ()
        
        
        # @var task_type string Determine if we are dealing with a regression, classification or 
        #                       multi-label problem
        task_type = self.dataset.get_task_type ()
        
        
        # @var model_folder String
        model_folder = self.get_folder ()
        
        
        # @var hyperparameters_file_path String
        hyperparameters_file_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'hyperparameters.csv')
        
        
        # @var resume_file_path String
        resume_file_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'training_resume.json')
        
        
        # Remove NaN for regression tasks
        # @note Alternative df['label'] = df['label'].fillna (0)
        if task_type == 'regression':
            df = df.dropna (subset = ['label'])
        
        
        # @var val_split String 
        # Determine which split do we get based on if we 
        # want to validate the val dataset of the train dataset
        val_split = 'val' if not using_official_test else 'test'
        
        
        # @var train_df DataFrame Get training split
        train_df = self.dataset.get_split (df, 'train')

        
        # If using the validation set for training
        if using_official_test:
            train_df = pd.concat ([train_df, self.dataset.get_split (df, 'val')])
            
        
        # @var val_df DataFrame Get validation split
        val_df = self.dataset.get_split (df, val_split)
        
        
        # @var language String 
        # Get the language of the dataset. This is useful when using 
        # pretrained word embeddings
        language = self.dataset.get_dataset_language ()

        
        # @var indexes Dict the the indexes for each split
        indexes = {split: subset.index for split, subset in {'train': train_df, 'val': val_df}.items ()}
        
        
        # @var available_labels List All the possible labels for classification and multi-label tasks
        available_labels = self.dataset.get_available_labels ()
        
        
        # @var is_imbalanced Boolean Determine if the dataset is imbalaced
        is_imbalanced = self.dataset.is_imbalanced ()
        
        
        # If the problem is classification, then we need to encode the label as numbers instead of using names
        if task_type in 'classification':
            
            # One hot encoding for multi-classification
            if self.dataset.get_num_labels () > 2:
            
                # Create a binarizer
                lb = sklearn.preprocessing.LabelBinarizer ()
                
                
                # Fit the label binarizer
                lb.fit (available_labels)
                
                
                # Note that we are dealing with one-hot enconding for multi-class
                train_df = pd.concat ([train_df, pd.DataFrame (lb.transform (train_df['label']), index = train_df.index, columns = lb.classes_)], axis = 1)
                val_df = pd.concat ([val_df, pd.DataFrame (lb.transform (val_df['label']), index = val_df.index, columns = lb.classes_)], axis = 1)

            
            # Encode labels as True|False for binary labels
            else:
                train_df['label'] = train_df['label'].astype ('category').cat.codes
                val_df['label'] = val_df['label'].astype ('category').cat.codes

        elif task_type in 'multi_label':

            # Create a binarizer
            lb = sklearn.preprocessing.MultiLabelBinarizer ()
            
            
            # Fit the multi-label binarizer
            lb.fit ([available_labels])
            
            
            # Transform as categories
            train_df['label'] = df['label'].astype ('category')
            val_df['label'] = df['label'].astype ('category')
            
            
            # Adding None category
            # @todo Move to another place, maybe?
            train_df['label'] = train_df['label'].cat.add_categories ('none')
            val_df['label'] = val_df['label'].cat.add_categories ('none')
            
            
            # Fill NaN values
            train_df['label'].fillna ('none', inplace = True)
            val_df['label'].fillna ('none', inplace = True)
            
            
            # Get all the labels for each instance. They are normally divided by ";"
            train_labels = [label for label in train_df['label'].str.split (';')]
            train_labels = [[y.strip () for y in list] for list in train_labels]

            val_labels = [label for label in val_df['label'].str.split (';')]
            val_labels = [[y.strip () for y in list] for list in val_labels]
            
            
            # Adjust the labels
            train_labels = lb.transform (train_labels)
            val_labels = lb.transform (val_labels)
            
            
            # Encode the labels and merge to the dataframe
            # Note we use "; " to automatically trim the texts
            train_df = pd.concat ([train_df, pd.DataFrame (train_labels, index = train_df.index, columns = lb.classes_)], axis = 1)
            val_df = pd.concat ([val_df, pd.DataFrame (val_labels, index = val_df.index, columns = lb.classes_)], axis = 1)


        # @var Tokenizer Tokenizer|None Only defined if we handle word embedding features
        tokenizer = None
        
        
        # @var maxlen int|None Only defined if we handle word embedding features
        maxlen = None
        
        
        # @var pretrained_word_embeddings List
        pretrained_word_embeddings = []

        
        # Generate the tokenizer employed 
        if 'we' in self.features:
        
            # Load the tokenizer from disk
            self.features['we'].load_tokenizer_from_disk (self.dataset.get_working_dir (self.dataset.task, 'we_tokenizer.pickle')) 
            
            
            # @var tokenizer Tokenizer
            tokenizer = self.features['we'].get_tokenizer ()
            
            
            # @var maxlen int Get maxlen
            maxlen = int (self.features['we'].maxlen)
            
            
            # @var pretrained_word_embeddings List Generate data and store them on cache
            pretrained_word_embeddings = ['fasttext', 'glove', 'word2vec'] if language == 'es' else ['fasttext']
            
            
            # Get embedding matrix
            for key in pretrained_word_embeddings:
                utils.get_embedding_matrix (
                    key = key, 
                    tokenizer = tokenizer, 
                    dataset = self.dataset,
                    lang = language
                )
                
                
        # Get the optimizers for hyper-parameter optimisation
        optimizers = [tensorflow.keras.optimizers.Adam]
        if task_type == 'regression':
            optimizers.append (tensorflow.keras.optimizers.RMSprop)
        
        
        # @var feature_combination Tuple
        feature_combination = self.get_feature_combinations ()
        
        
        # Get the reduction_metric according to the domain problem
        reduction_metric = 'val_loss'
        
        
        # Get parameters to evaluate
        params_epochs = [1000]
        params_lr = [10e-03, 10e-04]
        params_batch_size = [32, 64]
        params_dropout = [False, 0.1, 0.2, 0.3]
        params_pretrained_embeddings = ['none'] + pretrained_word_embeddings
        
        
        # With imbalaced datasets, we evaluate larger batch_sizes
        # Now create and train your model using the function that was defined earlier. Notice that the model is fit 
        # using a larger than default batch size of 2048, this is important to ensure that each batch has a 
        # decent chance of containing a few positive samples. If the batch size was too small, they would likely have 
        # no real data to learn
        # @link https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        # @todo Find heuristics to determine this size
        if is_imbalanced:
            params_batch_size = [128, 256, 512]
        
            
        # @var feature_key String
        feature_key = '-'.join (feature_combination)
        
        
        # @var architectures_to_evaluate List
        architectures_to_evaluate = ['dense', 'cnn', 'bilstm', 'bigru'] if 'we' in feature_key else ['dense']


        # However, it is possible to have a custom configuration
        # @todo Retrieve with a function and check if there is task 
        #       speific data
        if "custom_params" in dataset_options:
            if "deep_learning" in dataset_options['custom_params']:
                if "batch_size" in dataset_options['custom_params']['deep_learning']:
                    params_batch_size = dataset_options['custom_params']['deep_learning']['batch_size']

                if "lr" in dataset_options['custom_params']['deep_learning']:
                    params_lr = dataset_options['custom_params']['deep_learning']['lr']

                if "architectures_to_evaluate" in dataset_options['custom_params']['deep_learning'] and 'we' in feature_key:
                    architectures_to_evaluate = dataset_options['custom_params']['deep_learning']['architectures_to_evaluate']
        
        
        # @var best_model_file String
        best_model_file = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'model.h5')
        
        
        # If the file exists, then skip (unless we force to retrain)
        if os.path.isfile (best_model_file):
            if not force:
                print ("skiping " + best_model_file + ". Use --force=True to override")
                return
        
        
        # Skip merged datasets with pre-trained word embeddings
        if 'we' in feature_key and self.dataset.is_merged:
            print ("skiping we in a merged dataset")
            return
            
            
        # @var logs_path String
        logs_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'logs')
        
        
        # Delete previous path folder
        shutil.rmtree (self.dataset.get_working_dir (self.dataset.task, 'models', model_folder), ignore_errors = True)
        
        
        # @var main_metric String The metric that determines which is the best model
        main_metric = self.get_main_metric ()

        
        # @var initial_bias Float
        # @link http://karpathy.github.io/2019/04/25/recipe/
        # @todo Initial bias for multi-classification
        #
        # init well. Initialize the final layer weights correctly. E.g. if you are regressing some values that have a 
        # mean of 50 then initialize the final bias to 50. If you have an imbalanced dataset of a ratio 1:10 of 
        # positives:negatives, set the bias on your logits such that your network predicts probability of 0.1 at 
        # initialization. Setting these correctly will speed up convergence and eliminate “hockey stick” loss curves 
        # where in the first few iteration your network is basically just learning the bias.
        initial_bias = None
        if 'classification' == task_type and self.dataset.get_num_labels () <= 2:
        
            # @var train_label_counts int
            train_label_counts = train_df['label'].value_counts (sort = True).to_list ()
        
            
            # Set initial bias
            initial_bias = np.log ([train_label_counts[1] / train_label_counts[0]])
        
        
        # Class weight
        # Using class_weights changes the range of the loss and may affect the stability 
        # of the training depending on the optimizer. 
        # Optimizers whose step size is dependent on the magnitude of the gradient, 
        # like optimizers.SGD, may fail. The optimizer used here, optimizers.Adam, 
        # is unaffected by the scaling change. 
        # NOTE: Because of the weighting, the total losses are not comparable between the two models.
        # NOTE: Scaling keep the loss to a similar magnitude. The sum of the weights of all examples stays the same.
        # @link https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
        if 'regression' == task_type:
            class_weights = None
        
        
        if 'classification' == task_type:
            class_weights = dict (enumerate (class_weight.compute_class_weight (
                class_weight = 'balanced', 
                classes = available_labels, 
                y = self.dataset.get_split (df, 'train')['label']
            )))

        if 'multi_label' == task_type:
            class_weights = dict (enumerate (class_weight.compute_sample_weight (
                class_weight = 'balanced', 
                y = lb.transform (train_df['label'].str.split ('; ')) 
            )))
        
        
        # @var inputs_size Dict Get the input size for each feature set
        inputs_size = {
            key: pd.DataFrame (self.features[key].transform (df)).shape[1] if key in feature_key else 0 for key in self.all_available_features
        }
        
        
        # @var parameters_to_evaluate List
        # Note that some of these variables are not hyper-parameters
        # but only features we pass to the function that builds the 
        # Keras Models
        parameters_to_evaluate = []
        
        
        
        # Shallow neural networks
        if 'dense' in architectures_to_evaluate:
            parameters_to_evaluate.append ({
                'type': ['shallow'],
                'n_iter': [20],
                'architecture': ['dense'],
                'num_layers':  [1, 2],
                'shape': ['brick'],
                'activation': ['linear', 'relu', 'sigmoid', 'tanh']
            })
        
        
        # Deep neuronal networks
        if 'dense' in architectures_to_evaluate:
            parameters_to_evaluate.append ({
                'type': ['deep'],
                'n_iter': [5],
                'architecture': ['dense'],
                'num_layers': [3, 4, 5, 6, 7, 8],
                'shape': ['funnel', 'rhombus', 'lfunnel', 'brick', 'diamond', '3angle'],
                'activation': ['sigmoid', 'tanh', 'selu', 'elu'],
                'lr': [10e-03, 10e-04]
            })
        
        # Convolutional neuronal networks
        if 'cnn' in architectures_to_evaluate:
        
            # Convolutional neural networks
            parameters_to_evaluate.append ({
                'type': ['cnn'],
                'n_iter': [10],
                'architecture': ['cnn'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'kernel_size': [3, 4, 5],
                'activation': ['relu', 'tanh'],
                'first_neuron': [16, 32, 64],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })
            
            
        # Recurrent neuronal networks
        if 'gru' in architectures_to_evaluate:
            
            # Bidirectional Recurrent neuronal networks
            parameters_to_evaluate.append ({
                'type': ['gru'],
                'n_iter': [1],
                'architecture': ['gru'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'activation': ['relu'],
                'first_neuron': [4, 5],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })

        # Recurrent neuronal networks
        if 'lstm' in architectures_to_evaluate:
            
            # Bidirectional Recurrent neuronal networks
            parameters_to_evaluate.append ({
                'type': ['lstm'],
                'n_iter': [1],
                'architecture': ['lstm'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'activation': ['relu'],
                'first_neuron': [4, 5],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })
            
            
        # Recurrent neuronal networks
        if 'bigru' in architectures_to_evaluate:
            
            # Bidirectional Recurrent neuronal networks
            parameters_to_evaluate.append ({
                'type': ['bigru'],
                'n_iter': [5],
                'architecture': ['bigru'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'activation': ['relu'],
                'first_neuron': [4, 5],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })

        # Recurrent neuronal networks
        if 'bilstm' in architectures_to_evaluate:
            
            # Bidirectional Recurrent neuronal networks
            parameters_to_evaluate.append ({
                'type': ['bilstm'],
                'n_iter': [5],
                'architecture': ['bilstm'],
                'num_layers': [1, 2],
                'shape': ['brick'],
                'activation': ['relu'],
                'first_neuron': [4, 5],
                'batch_size': [max (params_batch_size)],
                'lr': [10e-03]
            })
        
        
        # the optimal size of the hidden layer is usually between the size of the input and size of the output layers
        # @link https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw
        # @var min_neurons int
        min_neurons = self.dataset.get_num_labels ()
        
        
        # @var max_neurons int
        max_neurons = sum (inputs_size.values ())

        
        # @var param_first_neuron List
        # @link https://www.researchgate.net/post/How-to-decide-the-number-of-hidden-layers-and-nodes-in-a-hidden-layer
        param_first_neuron = [neurons for neurons in [4, 8, 16, 48, 64, 128, 256, 512, 1024] if neurons >= min_neurons and neurons <= max_neurons]
        param_first_neuron.append (min_neurons)
        param_first_neuron.append (10 + int (pow (round (min_neurons + max_neurons), .5)))
        param_first_neuron = list (set (param_first_neuron))
        
        
        # Common parameters for all groups (except if they were defined before)
        for group in parameters_to_evaluate:
            
            if 'optimizer' not in group:
                group['optimizer'] = optimizers
                
            if 'first_neuron' not in group:
                group['first_neuron'] = param_first_neuron
            
            if 'lr' not in group:
                group['lr'] =  params_lr

            if 'epochs' not in group:
                group['epochs'] = params_epochs
                
            if 'batch_size' not in group:
                group['batch_size'] = params_batch_size
                
            if 'dropout' not in group:
                group['dropout'] = params_dropout


        # Calculate the probability of each trials
        n_trials = 0
        for group in parameters_to_evaluate:
            n_trials = n_trials + group['n_iter'][0]

        for group in parameters_to_evaluate:
            group['probability'] = group['n_iter'][0] / n_trials
        
        
        # @var scheduler Use HyperBand scheduler to earlystop unpromising runs
        scheduler = ray.tune.schedulers.AsyncHyperBandScheduler (time_attr = 'training_iteration',
            metric = main_metric,
            mode = 'max' if task_type == 'classification' else 'min',
            grace_period = 10
        )
        
        
        # @var x Dict of features for each subset
        x = {}
        for subset in ['train', 'val']:
            x[subset] = {}
            for key in feature_combination:
                features = pd.DataFrame (self.features[key].transform (df))
                
                """
                # Uncomment this for testing "input-indepent baseline"
                # This changes all the inputs features to 0
                # The idea is that the network learn worse than with input data
                # Do not worry if the models learns, it is becasue the "bias"
                # You could change it as: keras.layers.Dense (use_bias = False)
                # @link http://karpathy.github.io/2019/04/25/recipe/#2-set-up-the-end-to-end-trainingevaluation-skeleton--get-dumb-baselines
                for col in features.columns:
                    features[col].values[:] = 0
                """
                
                x[subset]['input_' + key] = tensorflow.cast (features[features.index.isin (indexes[subset])].reindex (indexes[subset]), tensorflow.float32)
                

        
        # @var y_labels_columns List
        y_labels_columns = lb.classes_ if self.dataset.get_num_labels () > 2 else 'label'
        
        
        # @var y labels
        y = {
            'train': tensorflow.convert_to_tensor (train_df[y_labels_columns].values),
            'val': tensorflow.convert_to_tensor (val_df[y_labels_columns].values)
        }
        
        
        # @var train_resume Dict We track information of the training process
        # including the features, best model, metrics, etc.
        train_resume = {
            'model': 'deep-learning',
            'folder': model_folder,
            'dataset': self.dataset.dataset,
            'corpus': self.dataset.corpus,
            'task': self.dataset.task,
            'task_type': task_type,
            'initial_bias': initial_bias.tolist () if initial_bias else "",
            'main_metric': main_metric,
            'features': {key: self.features[key].cache_file for key in feature_combination}
        };
        
        
        if task_type in ['classification']:
            train_resume['train_labels'] = train_df['label'].value_counts ().to_json ()
            train_resume['val_labels'] = val_df['label'].value_counts ().to_json ()
            train_resume['class_weights'] = str (json.dumps (class_weights))

        
        # Attach classification stuff to the resume
        # @todo Change train labels for multi_label problems
        if task_type in ['classification', 'multi_label']:
            # @var f1_score_scheme String
            train_resume['f1_score_scheme'] = dataset_options['f1_score_scheme'] \
                if 'f1_score_scheme' in dataset_options else 'micro'
            
        
        # Start logging
        print ("dataset {dataset} corpus {corpus} task {task}".format (
            dataset = train_resume['dataset'], 
            corpus = train_resume['corpus'],
            task = train_resume['task']
        ))
        
        
        # Print resume
        print ('resume')
        print ('------')
        print (json.dumps (train_resume, indent = 4))
        
        
        def ray_train (config, dataset, features, x, y, class_weights, reporter = None, checkpoint_dir = None, data_dir = None):
                
            def get_learning_rate_scheduler_callback (lr, epochs):
                """
                @link https://towardsdatascience.com/learning-rate-schedule-in-practice-an-example-with-keras-and-tensorflow-2-0-2f48b2888a0c
                """
                
                # @var decay long
                decay = lr / epochs
                def lr_time_based_decay (epoch, lr):
                    return lr * 1 / (1 + decay * epoch)
                    
                
                # @var lr_scheduler LearningRateScheduler
                return tensorflow.keras.callbacks.LearningRateScheduler (lr_time_based_decay)


            def get_y_pred (model, X):
            
                # @var task_type
                task_type = dataset.get_task_type ()
                
                
                # @var labels 
                labels = dataset.get_available_labels ();
            
                
                # @var y_pred Predict 
                y_pred = model.predict (X, batch_size = len (X))
                
                
                # Transform predictions into binary or multiclass
                if 'classification' == task_type:
                    y_pred = y_pred > .5 if dataset.get_num_labels () <= 2 else np.argmax (y_pred, axis = 1)
                    return [labels[int (item)] for item in y_pred]
                
                elif 'multi_label' == task_type:
                    return y_pred.round ()
                    
                
            def get_y_real (tensorflow_dataset):
                
                # @var task_type
                task_type = dataset.get_task_type ()
                
                
                # @var labels 
                labels = dataset.get_available_labels ();
                
                
                # Transform predictions into binary or multiclass
                if 'classification' == task_type:
                    
                    # @var y_real 
                    y_real = np.concatenate ([y for x, y in tensorflow_dataset], axis = 0)
                    y_real = y_real > .5 if dataset.get_num_labels () <= 2 else np.argmax (y_real, axis = 1)
                    return [labels[int (item)] for item in y_real]
                
                
                if 'multi_label' == task_type:
                    return np.concatenate ([y for x, y in tensorflow_dataset], axis = 0)


            def get_early_stopping_metric ():
                
                # @var task_type String
                task_type = dataset.get_task_type ()
                
                
                # Regression metrics
                if 'regression' == task_type:
                    return 'val_rmse'
                
                # Regression metrics
                if 'classification' == task_type:
                    return 'val_prc' if dataset.is_imbalanced () else 'val_loss'
                
                # Multi_label metrics
                if 'multi_label' == task_type:
                    return 'val_loss'


            def get_patience_per_achitecture (architecture, features):
                
                # @var patience int For the early stoppping mechanism
                patience = 100
                
                
                if 'we' in features:
                    patience = 10
                
                if architecture in ['lstm', 'gru', 'bilstm', 'bigru']:
                    patience = 5
                    
                if architecture in ['cnn']:
                    patience = 10
                    
                return patience

            def get_early_stopping_callback (patience):
                
                # @var early_stopping Early Stopping callback
                # @todo Note this: https://github.com/tensorflow/tensorflow/issues/35634
                return tensorflow.keras.callbacks.EarlyStopping (
                    monitor = get_early_stopping_metric (), 
                    patience = patience,
                    mode = 'auto',
                    restore_best_weights = False
                )

                
            # Reset graph
            tensorflow.compat.v1.reset_default_graph ()
            tensorflow.keras.backend.clear_session ()


            # @var task_type
            task_type = dataset.get_task_type ()

            
            # @var my_config Dict
            my_config = {key: value for key, value in config['type'].items ()}
            
            
            # Insert all features at once
            my_config['features'] = features
            
            
            # Set seed for this model (if specified)
            np.random.seed (my_config['seed'])
            
            
            # @var weights_file String
            weights_file = tempfile.mkdtemp ()
            
            
            # @var lr_scheduler LearningRateScheduler
            lr_scheduler = get_learning_rate_scheduler_callback (
                lr = my_config['lr'],
                epochs = global_model_data['epochs']
            )
            
            
            # @var early_stopping Early Stopping callback
            early_stopping = get_early_stopping_callback (
                patience = get_patience_per_achitecture (my_config['architecture'], my_config['features'])
            )
            
            
            # @var checkpoint_callback ModelCheckpoint
            checkpoint_callback = tensorflow.keras.callbacks.ModelCheckpoint (
                filepath = weights_file,
                monitor = main_metric, 
                save_best_only = True,
                save_weights_only = True,
                mode = 'max' if task_type == 'classification' else 'min',
                verbose = 0
            )
            
            
            # @var datasets Dict
            # @link https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle/48096625#48096625
            datasets = {split: tensorflow.data.Dataset.from_tensor_slices ((x[split], y[split])) \
                .cache () \
                .shuffle (len (y[split]), reshuffle_each_iteration = False) \
                .batch (my_config['batch_size']) \
                    for split in ['train', 'val']}
            
            
            # @var model Keras model based on the parameters
            model = kerasModel.create (my_config, dataset, inputs_size)
            
            
            # Create callbacks to be used during model training
            callbacks = [
                lr_scheduler,
                early_stopping,
                checkpoint_callback
            ]
            
            
            # @var dataset_options Dict
            dataset_options = dataset.get_task_options ()
            
            
            # @var history
            history = model.fit (
                x = datasets['train'], 
                validation_data = datasets['val'],
                epochs = global_model_data['epochs'],
                callbacks = callbacks,
                verbose = 0,
                class_weight = class_weights
            )
            
            
            # The model weights (that are considered the best) are loaded again into the model.
            model.load_weights (weights_file)
            
            
            # Remove the temporal directory
            shutil.rmtree (weights_file)


            # Save evaluated model
            with tune.checkpoint_dir (step = 1) as checkpoint_dir:
                model.save (os.path.join (checkpoint_dir, 'model.h5'))

            
            # Save the best model architecture
            with open (os.path.join (checkpoint_dir, 'model.json'), 'w') as f:
                with redirect_stdout (f):
                    model.to_json ()

            
            # @var task_type
            task_type = dataset.get_task_type ()
            
            
            # Get results for this model
            if task_type in ['classification', 'multi_label']:
            
                # @var validation_predictions 
                validation_predictions = get_y_pred (model, datasets['val'])
                
                
                # @var train_predictions 
                train_predictions = get_y_pred (model, datasets['train'])
                
                
                # @var validation_real 
                validation_real = get_y_real (datasets['val'])
                
                
                # @var train_real 
                train_real = get_y_real (datasets['train'])
                
                
                # @var f1_score_scheme String
                f1_score_scheme = dataset_options['f1_score_scheme'] \
                    if 'f1_score_scheme' in dataset_options else 'micro'
                
                
                tune.report (
                    loss = history.history['loss'][-1],
                    val_loss = history.history['val_loss'][-1],
                    accuracy = accuracy_score (y_true = train_real, y_pred = train_predictions),
                    val_accuracy = accuracy_score (y_true = validation_real, y_pred = validation_predictions),
                    f1_score = f1_score (y_true = train_real, y_pred = train_predictions, average = f1_score_scheme),
                    val_f1_score = f1_score (y_true = validation_real, y_pred = validation_predictions, average = f1_score_scheme),
                )
                
            
            # For regression
            if 'regression' == task_type:
            
                # Training
                metrics_results = model.evaluate (datasets['train'], verbose = 0, batch_size = len (train_df)) 
                for index, metric in enumerate (model.metrics_names):
                    model_results[metric] = metrics_results[index]
            
                
                # Validation
                metrics_results = model.evaluate (datasets['val'], verbose = 0, batch_size = len (val_df)) 
                for index, metric in enumerate (model.metrics_names):
                    model_results['val_' + metric] = metrics_results[index]
                    
        
        
        # @var global_model_data Dict
        global_model_data = {
            "epochs": parameters_to_evaluate[0]['epochs'][0],
        }
        
        
        # @var parameter_choices List Default and common values
        parameter_choices = []
        
        
        # ...
        for group in parameters_to_evaluate:
        
            # @var type String
            type = group['type'][0]
            
            
            # @var probability Float
            probability = group['probability']
            
            
            # @var parameters_to_add Dict 
            parameters_to_add = {
                'type': hp.choice (type + '_network', [type]),
                'seed': hp.randint (type + '_seed', 0, 1000),
                'pretrained_embeddings': hp.choice (type + '_pretrained_embeddings', params_pretrained_embeddings),
                'architecture': hp.choice (type + '_architecture', group['architecture']),
                'num_layers': hp.choice (type + '_num_layers', group['num_layers']),
                'activation': hp.choice (type + '_activation', group['activation']),
                'first_neuron': hp.choice (type + '_first_neuron', group['first_neuron']),
                'batch_size': hp.choice (type + '_batch_size', group['batch_size']),
                'dropout': hp.choice (type + '_dropout', group['dropout']),
                'optimizer': hp.choice (type + '_optimizer', group['optimizer']),
                'shape': hp.choice (type + '_shape', group['shape']),
                'lr': hp.choice (type + '_lr', group['lr'])
            }
            
            
            # Conditional fields
            if initial_bias:
                parameters_to_add['output_bias'] = hp.choice (type + "_output_bias", initial_bias)

            if 'kernel_size' in group:
                parameters_to_add['kernel_size'] = hp.choice (type + "_kernel_size", group['kernel_size'])

            # Attach word embeddings stuff
            if 'we' in feature_combination:
                parameters_to_add['tokenizer'] = hp.choice (type + "_tokenizer", [tokenizer])
                parameters_to_add['maxlen'] = hp.choice (type + "_maxlen", [maxlen])
            
            
            # Attach
            # parameter_choices.append ((probability, parameters_to_add))
            parameter_choices.append (parameters_to_add)


        # @var search_space Dict
        search_space = {
            "type": hp.choice ('type', parameter_choices)
        }
        

        # @var search_alg Use bayesian optimisation with TPE implemented by hyperopt
        search_alg = ray.tune.suggest.hyperopt.HyperOptSearch (search_space,
            metric = main_metric, 
            mode = 'max' if task_type == 'classification' else 'min'
        )


        # We limit concurrent trials to 2 since bayesian optimisation doesn't parallelize very well
        search_alg = ray.tune.suggest.ConcurrencyLimiter (search_alg, max_concurrent = 2)
        
        
        # Limit the number of rows.
        reporter = CLIReporter (max_progress_rows = 10)
        reporter.add_metric_column ("f1_score")
        reporter.add_metric_column ("val_f1_score")


        @ray.remote
        class HyperparametersDataFrame:
            
            # @var hyperparameter_df None
            hyperparameter_df = None
            
            def add (self, row):
                
                # @var temp_df DataFrame
                temp_df = pd.DataFrame.from_records ([row], index = 'trial_id')

                # Attach the row
                if self.hyperparameter_df is None:
                    self.hyperparameter_df = temp_df
                
                else:
                    self.hyperparameter_df = pd.concat ([self.hyperparameter_df, temp_df])

            def update_metric (self, trial_id, metric, objective):
                self.hyperparameter_df.loc[trial_id, metric] = objective

            def get (self):
                return self.hyperparameter_df
        
        
        # @var hyperparameter_df_handler 
        hyperparameter_df_handler = HyperparametersDataFrame.remote ()
        
        
        class GetBestTrialCallback (Callback):
            """
            GetBestTrialCallback
            """
            
            def __init__ (self, hyperparameter_df_handler):
                self.hyperparameter_df_handler = hyperparameter_df_handler
            
            
            def on_trial_start (self, iteration, trials, trial, **info):

                # @var trial_parameters Dict
                trial_parameters = {}
                
                
                # Set parameters
                trial_parameters['trial_id'] = trial.trial_id
                trial_parameters['objective'] = 0
                trial_parameters['best'] = False
                trial_parameters = {**trial_parameters, **trial.evaluated_params}

                
                # Attach
                self.hyperparameter_df_handler.add.remote (trial_parameters)
                
            
            def on_trial_result (self, iteration, trials, trial, result, **info):
                if 'done' in result:
                
                
                    # Update the result in the global dataframe
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'objective', result[main_metric])
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'loss', result['loss'])
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'val_loss', result['val_loss'])
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'accuracy', result['accuracy'])
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'val_accuracy', result['val_accuracy'])
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'time_this_iter_s', result['time_this_iter_s'])
                    
                    
                    # Report result
                    print ("Iteration {iteration} finished. Result: {objective}".format (
                        iteration = iteration, 
                        objective = result[main_metric]
                    ))
                    
                    
        # @var resources_per_trial Dict
        resources_per_trial = {
            "cpu": min (8, multiprocessing.cpu_count ())
        }
        
        
        # @var callback GetBestTrialCallback
        callback = GetBestTrialCallback (hyperparameter_df_handler)

        
        # Run the hyperparameter tunning
        # The usage of with_parameters is needed in order to avoid 
        # memory restrictions
        # @link https://docs.ray.io/en/latest/tune/user-guide.html
        analysis = tune.run (
            tune.with_parameters (ray_train, 
                dataset = self.dataset, 
                x = x, 
                y = y, 
                class_weights = class_weights, 
                features = self.features
            ),
            local_dir = self.dataset.get_working_dir (self.dataset.task, 'models'),
            name = model_folder,
            sync_config = tune.SyncConfig (syncer = None),
            verbose = 0, 
            num_samples = n_trials,
            search_alg = search_alg,
            scheduler = scheduler,
            fail_fast = True,
            raise_on_failed_trial = False,
            progress_reporter = reporter,
            resources_per_trial = resources_per_trial,
            callbacks = [callback]
        )


        # @var hyperparameter_df DataFrame
        hyperparameter_df = ray.get (hyperparameter_df_handler.get.remote ()).copy (deep = True)
        
        
        # @var best_run_id String
        best_run_id = hyperparameter_df['objective'].idxmax ()
        
        
        # Store the maximum value
        hyperparameter_df.loc[best_run_id, 'best'] = True
        
        
        # Show hyperparameters
        print (hyperparameter_df)
        hyperparameter_df.to_csv (hyperparameters_file_path, index = False)
        
        
        # Store the training resume
        with open (resume_file_path, 'w') as resume_file:
            json.dump (train_resume, resume_file, indent = 4, sort_keys = True)        
        
        
        # Remove unused folders
        p = Path (self.dataset.get_working_dir (self.dataset.task, 'models', model_folder))
        for directory in [x for x in p.iterdir () if x.is_dir ()]:
            
            # Keep the best model
            if best_run_id in str (directory):
                shutil.copyfile (os.path.join (directory, 'checkpoint_000001', 'model.h5'), self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'model.h5'))
            
            
            # Remove unwanted stuff
            shutil.rmtree (directory, ignore_errors = True)  

    def get_best_model (self, model_folder, use_train_val = False):
        """
        @inherit
        """
        
        # @var model_index int|None Used to select the best model according to a custom criteria
        model_index = None
        
        
        # Get the best model based on some criteria
        if self.best_model_criteria:
            
            # @var hyperparameter_df DataFrame
            hyperparameter_df = pd.read_csv (self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'hyperparameters.csv'))
            hyperparameter_df = hyperparameter_df.rename (columns = {'Unnamed: 0': 'index'})

            
            # @var filter String Clone the original criteria and attach filter to get the best
            filter = self.best_model_criteria
            
            
            # Select a subframe
            # @link https://stackoverflow.com/questions/34157811/filter-a-pandas-dataframe-using-values-from-a-dict
            hyperparameter_df = hyperparameter_df.loc[(hyperparameter_df[list (filter)] == pd.Series (filter)).all (axis = 1)]

            
            # @var main_metric get metric
            main_metric = self.get_main_metric ()
            
            
            # Update model index
            model_index = hyperparameter_df.sort_values (by = main_metric, ascending = True).tail (1)[main_metric].index.item ()
            
            
        # @var model_name Model name according to criteria
        if use_train_val:
            model_name = 'model_with_val.h5'
            
        elif model_index is not None:
            model_name = 'model-' + str (model_index) + '.h5'
            
        else:
            model_name = 'model.h5'
            
        
        # @var model_filename String Retrieve the filepath of the best model
        model_filename = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, model_name)
        
        
        # Ensure the model exists. If not, return nothing
        if not os.path.exists (model_filename):
            return
    
    
        # @var best_model KerasModel
        try:
            return tensorflow.keras.models.load_model (model_filename, compile = False)
        
        except Exception as e:
            print ("model could not be loaded. Skip...")
            print (model_filename)
            print (e)
            print ("......")
            return
            

    def predict (self, using_official_test = False, callback = None, use_train_val = False):
        """
        @inherit
        """
        
        # @link https://stackoverflow.com/questions/58814130/tensorflow-2-0-custom-keras-metric-caused-tf-function-retracing-warning/62298471#62298471
        # tensorflow.compat.v1.disable_eager_execution ()
        
        
        # @var feature_combination Tuple
        feature_combination = self.get_feature_combinations ()
        
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var task_type string Determine if we are dealing with a regression or classification problem
        task_type = self.dataset.get_task_type ()
        
        
        # @var model_folder String
        model_folder = self.get_folder ()


        # @var hyperparameters_file_path String
        hyperparameters_file_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'hyperparameters.csv')
        
        
        # @var hyperparameter_df DataFrame 
        hyperparameter_df = pd.read_csv (hyperparameters_file_path)
        
        
        # Maybe this step is not necessary, as random numbers are not generated 
        # during prediction
        if 'type/seed' in hyperparameter_df.columns:
        
            # @var seed int
            seed = int (hyperparameter_df.loc[hyperparameter_df['best'] == True]['type/seed'])
        
        
            # Set seed
            np.random.seed (seed)
        
        
        # @var resume_file_path String
        resume_file_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'training_resume.json')
        
        
        # @var best_model KerasModel
        best_model = self.get_best_model (model_folder, use_train_val = use_train_val)
                
        
        # @var x Dict of features for each subset
        x = {key: pd.DataFrame (self.features[key].transform (df)) for key in feature_combination}
        
        
        # If the supplied dataset contains information of the split, that means that we are dealing
        # only with a subset of the features and we retrieve it accordingly
        if using_official_test:
            x = {'input_' + key: item[item.index.isin (df.index)].reindex (df.index) for key, item in x.items ()}
        
        else:
            x = {'input_' + key: item for key, item in x.items ()}
        
        
        # @var raw_predictions Get the logits of the predictions
        raw_predictions = best_model.predict (x)
        
        
        # According to the number of labels we discern between binary and multiclass
        if 'classification' == task_type:
            if self.dataset.get_num_labels () <= 2:
                predictions = raw_predictions >= 0.5
                predictions = np.squeeze (predictions)
                
                raw_predictions = [[float (prediction), float (1 - prediction)] for prediction in raw_predictions]
            
            # Multiclass
            else:
                predictions = np.argmax (raw_predictions, axis = 1)
        
        
            # @var true_labels
            true_labels = self.dataset.get_available_labels ()
            
            
            # @var predictions List
            y_pred = [true_labels[int (prediction)] for prediction in predictions]
            
        
        
        # Multi label tasks
        if 'multi_label' == task_type:
            
            # Round the predictions
            predictions = raw_predictions
            
            
            # @var y_pred List
            y_pred = predictions.round ()
            

        
        # Regression tasks
        if 'regression' == task_type:
            
            # @var predictions Are the raw predictions
            predictions = raw_predictions
        
            
            # @var y_pred List
            y_pred = predictions.squeeze ()
        
        
        # @var model_metadata Dict
        model_metadata = {
            'model': best_model,
            'probabilities': raw_predictions
        }
        
        
        # run callback
        if callback:
            callback (model_folder, y_pred, model_metadata)
    
    

   