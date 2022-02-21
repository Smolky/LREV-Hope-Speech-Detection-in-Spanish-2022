"""
    To do random stuff
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import config
import argparse
import pandas as pd
import numpy as np
import pickle
import re
from pathlib import Path

import transformers
import shap

from tqdm import tqdm
from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser
import tensorflow


def main ():

    tensorflow.compat.v1.disable_eager_execution ()


    # var parser
    parser = DefaultParser (description = 'To do random stuff')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # Add model
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family of algorithms to evaluate', 
        choices = model_resolver.get_choices ()
    )
    
    
    # @var choices List of list 
    choices = FeatureResolver.get_feature_combinations (['lf', 'se', 'be', 'we', 'ne', 'cf', 'bf', 'pr'])
    
    
    # Add features
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to evaluate', 
        choices = ['all'] + ['-'.join (choice) for choice in choices]
    )
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()
    
    
    # @var model Model
    model = model_resolver.get (args.model)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()    
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var feature_combinations List
    feature_combinations = get_feature_combinations (model.get_available_features ()) if args.features == 'all' else [args.features.split ('-')]
    
    
    # Load all the available features
    for features in feature_combinations:
            
        # Indicate which features we are loading
        print ("loading features...")
        
        for feature_set in features:

            # @var feature_file String
            feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)

        
            # @var features_cache String The file where the features are stored
            features_cache = dataset.get_working_dir (args.task, feature_file)

            
            # If the feautures are not found, get the default one
            if not Path (features_cache, cache_file = "").is_file ():
                features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')
            
            
            # Indicate what features are loaded
            print ("\t" + features_cache)
            if not Path (features_cache).is_file ():
                print ("skip...")
                continue
            
            
            # Set features
            model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))
    
    
    # @var feature_combination Tuple
    feature_combination = model.get_feature_combinations ()
    
    
    # @var feature_key String
    feature_key = '-'.join (feature_combination)
    
    
    # Predict this feature set
    best_model = model.get_best_model (feature_key)
    
    
    # @var train_df DataFrame Get training split
    train_df = dataset.get_split (df, 'train')


    # @var test_df DataFrame Get test split
    test_df = dataset.get_split (df, 'test')

    
    # @var 
    transformer = model.get_features ('lf')
    
    
    features = pd.DataFrame (transformer.transform (df))
    x_train = features[features.index.isin (train_df.index)].reindex (train_df.index)
    x_test = features[features.index.isin (test_df.index)].reindex (test_df.index)
    

    # @var background
    background = dataset.get ()['label']
    
    
    # explain the model on two sample inputs
    explainer = shap.DeepExplainer (best_model, x_train) 
    
    
    # @var x_train_labels
    x_train_labels = df.loc[x_train.index]['label']
    

    # @var shap_values
    shap_values = explainer.shap_values (x_train.head (200).values)
    

    # @var data
    data = shap.force_plot (explainer.expected_value[0], shap_values[0], x_test)

    shap.save_html (dataset.get_working_dir (args.task, 'test.html'), data)
    
    
    
    
    
    def f (X):
        return best_model.predict (X)
    
    
    explainer = shap.KernelExplainer (f, x_train.iloc[:50, :])
    shap_values = explainer.shap_values (x_test.iloc[0].values, nsamples = 500)
    data = shap.force_plot (explainer.expected_value, shap_values[1], x_train_labels.iloc[0])
    shap.save_html (dataset.get_working_dir (args.task, 'test2.html'), data)

    """
    data = shap.summary_plot (shap_values, x_test)

    shap.save_html (dataset.get_working_dir (args.task, 'test2.html'), data)
    """
    
    """
    vals = np.abs (shap_values).mean (0)
    feature_importance = pd.DataFrame(list (zip (features.columns, sum (vals))), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values (by = ['feature_importance_vals'], ascending = False, inplace = True)
    print (feature_importance.head (10))
    """
    
    sys.exit ()
    
    
    # Clear session
    model.clear_session ();
    
    # @var feature_combinations List
    feature_combinations = get_feature_combinations (model.get_available_features ()) if args.features == 'all' else [args.features.split ('-')]
    
    
    # @var huggingface_model String
    huggingface_model = dataset.get_working_dir (dataset.task, 'models', 'bert', 'bert-finetunning')
    
    
    # load a transformers pipeline model
    model = transformers.pipeline ('sentiment-analysis', model = huggingface_model, return_all_scores = True)
    
    
    # explain the model on two sample inputs
    explainer = shap.Explainer (model) 
    shap_values = explainer (df.head (10)['tweet_clean_lowercase'])
    
    
    
    # visualize the first prediction's explanation for the POSITIVE output class
    # Exception: In v0.20 force_plot now requires the base value as the first parameter! 
    # Try shap.force_plot(explainer.expected_value, shap_values) or 
    # for multi-output models try 
    # shap.force_plot(explainer.expected_value[0], shap_values[0]).
    print (df.head (10)['tweet_clean_lowercase']) 
    

    # data = shap.force_plot (shap_values[0, :, 1])
    # shap.save_html (dataset.get_working_dir (args.task, 'test.html'), data)
    
    
    data = shap.plots.text (shap_values[0])
    print (data)
    shap.save_html (dataset.get_working_dir (args.task, 'test-2.html'), data)
    
        
    
if __name__ == "__main__":
    main ()
