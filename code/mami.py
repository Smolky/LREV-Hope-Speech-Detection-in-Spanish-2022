"""
    MAMI output generation
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import csv
import json
import sklearn
import itertools
import re
from zipfile import ZipFile

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser


def main ():

    # @var dataset_name String
    dataset_name = 'mami'
    
    
    # @var corpus_name String
    corpus_name = '2021'


    # var parser
    parser = DefaultParser (description = 'MAMI Submission')


    # Add features
    parser.add_argument ('--test', 
        dest = 'test', 
        default = 'run-01', 
        help = 'Determines the run', 
    )
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var tests Dict
    tests = {
        'run-01': {'folder': 'deep-learning-bf'},
        'run-02': {'folder': 'deep-learning-lf'},
        'run-03': {'folder': 'deep-learning-se'},
        'run-04': {'folder': 'deep-learning-we'},
        'run-05': {'folder': 'deep-learning-it'},
        
        'run-06': {'folder': 'ensemble-all-features-mode'},
        'run-07': {'folder': 'ensemble-all-features-highest'},
        'run-08': {'folder': 'ensemble-all-features-mean'},
        
        'run-09': {'folder': 'deep-learning-all-features'},
        'run-10': {'folder': 'bert-unbalanced'},
        'run-11': {'folder': 'deep-learning-lf-bf-it'},             # Unbalanced
        'run-12': {'folder': 'deep-learning-bf-it'},                # Unbalanced
        'run-13': {'folder': 'deep-learning-lf-bf'},                # Unbalanced
        'run-14': {'folder': 'deep-learning-bi'},                   # Unbalanced
        'run-15': {'folder': 'ensemble-lf-bf-bi-mean'},
        'run-16': {'folder': 'deep-learning-bf-bi'},                # Unbalanced
        'run-17': {'folder': 'deep-learning-lf-bf-bi-roberta'},
        'run-18': {'folder': 'deep-learning-lf-bf-bi'},             # Unbalanced
        'run-19': {'folder': 'deep-learning-lf-bf-bi-rf'}           # Unbalanced
        
    }


    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var test Dict
    test = tests[args.test]
    
    
    # @var source String
    source = 'test'
    
    
    # @var folder String
    folder = test['folder']


    # @var test_df DataFrame
    test_df = pd.DataFrame ()


    # @var model_values Dict
    model_values = {}

    
    # Iterate tasks
    for task_name in ['task-1', 'task-2-shaming', 'task-2-stereotype', 'task-2-objectification', 'task-2-violence']:

        # @var label_name String
        if task_name == 'task-1':
            label_name = 'misogynous'
        else:
            label_name = re.sub (r'task-[12]-', '', task_name)
        

        # Specify the rest of the args
        args.dataset = dataset_name
        args.corpus = corpus_name
        args.task = task_name
        
        
        # @var dataset_resolver DatasetResolver
        dataset_resolver = DatasetResolver ()
        
        
        # @var dataset Dataset This is the custom dataset for evaluation purposes
        dataset = dataset_resolver.get (dataset_name, corpus_name, task_name, False)
        
        
        # Determine if we need to use the merged dataset or not
        dataset.filename = dataset.get_working_dir (task_name, 'dataset.csv')
    
    
        # @var training_resume_file String
        training_resume_file = dataset.get_working_dir (dataset.task, 'models', folder, 'training_resume.json')
        
        
        with open (training_resume_file) as json_file:
            training_resume = json.load (json_file)


        # @var model_type String
        model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
        
        
        # @var model Model
        model = model_resolver.get (model_type)
        model.set_folder (folder)
        model.set_dataset (dataset)
        model.is_merged (dataset.is_merged)
    
        if model_type == 'transformers':
            model.set_pretrained_model (training_resume['pretrained_model'])
        
        if model_type == 'ensemble':
            model.set_ensemble_strategy (training_resume['strategy'])
    
    
        # @var task_type String
        task_type = dataset.get_task_type ()


        # @var feature_resolver FeatureResolver
        feature_resolver = FeatureResolver (dataset)
        
        
        # @var df DataFrame
        df = dataset.get ()
        
        
        # Replace the dataset to contain only the test or val-set
        if source in ['train', 'val', 'test']:
            dataset.default_split = source


        # @var df Ensure if we already had the data processed
        df = dataset.get ()
        
        
        # @var feature_combinations List
        feature_combinations = training_resume['features'] if 'features' in training_resume else {}


        # Prepare df_test
        if 'twitter_id' not in test_df.columns:
            test_df = df[['twitter_id']]


        def callback (feature_key, y_pred, model_metadata):
            model_values[task_name] = [1 if item == label_name else 0 for item in y_pred]


        # Load all the available features
        for feature_set, features_cache in feature_combinations.items ():
            
            # Indicate what features are loaded
            print ("\t" + features_cache)
            if not Path (features_cache).is_file ():
                print ("skip...")
                continue
            
            
            # Set features
            model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))

        
        
        # Predict this feature set
        model.predict (using_official_test = True, callback = callback)
        
        
        # Clear session
        model.clear_session ();


    # Set the results
    for key, values in model_values.items ():
        test_df[key] = values
        

    # @var answer_path String
    answer_path = dataset.get_working_dir ('runs', args.test, 'answer.txt')


    # Store
    test_df.to_csv (answer_path, header = False, index = False, sep = '\t', quoting = csv.QUOTE_NONE)

    
    # @var zip_file_path String
    zip_file_path = dataset.get_working_dir ('runs', args.test, args.test + '.zip')
    
    
    with ZipFile (zip_file_path, 'w') as zipf:
        zipf.write (answer_path, arcname = 'answer.txt')


if __name__ == "__main__":
    main ()
