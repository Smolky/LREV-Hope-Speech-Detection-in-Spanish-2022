"""
    Evaluate a new text or a test dataset
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import csv
import sklearn

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser
from pipeline.Tagger import Tagger
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.PrettyPrintConfussionMatrix import PrettyPrintConfussionMatrix


def main ():

    # var parser
    parser = DefaultParser (description = 'Evaluate EMOEVAL')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # @var confussion_matrix_pretty_printer PrettyPrintConfussionMatrix
    confussion_matrix_pretty_printer = PrettyPrintConfussionMatrix ()
    
    
    # Add model
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family of algorithms to evaluate', 
        choices = model_resolver.get_choices ()
    )
    
    
    # Add features
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to evaluate', 
        choices = ['all', 'lf', 'se', 'be', 'we', 'ne', 'pr', 'lf-bf']
    )
    
    
    # Add features
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to evaluate', 
        choices = ['all', 'train', 'test', 'val']
    )
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
        
    
    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var model Model
    model = model_resolver.get (args.model)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)


    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)

    
    # Replace the dataset to contain only the test or val-set
    if args.source in ['train', 'val', 'test']:
        dataset.default_split = args.source

    
    # @var available_features List
    available_features = model.get_available_features () if args.features == 'all' else [args.features]
    available_features = ['lf', 'bf']
    
    
    # Iterate over all available features
    for feature_set in available_features:

        # @var feature_file String
        feature_file = feature_resolver.get_suggested_cache_file (feature_set)

    
        # @var features_cache String The file where the features are stored
        features_cache = dataset.get_working_dir (args.task, feature_file)
        
        
        # Get default features
        if not Path (features_cache).is_file ():
            features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')

        
        # @var transformer
        transformer = feature_resolver.get (feature_set, cache_file = features_cache)
        
    
        # Set the features in the model
        model.set_features (feature_set, transformer)
    
    
    def callback (feature_key, y_pred, model_metadata):
        
        
        # @var labels List
        labels = dataset.get_available_labels ()
        
        
        # @var result DataFrame
        result = pd.DataFrame ({
            'id': dataset.get ()['twitter_id'],
            'y_pred': y_pred
        })
        
        print (result)
        print (result['y_pred'].value_counts ())
        
        
        # @var filename String
        filename = dataset.get_working_dir (args.task, 'umuteam_' + args.source + '_' + feature_key + '.tsv')
        
        
        # Store
        result.to_csv (filename, header = False, sep="\t", quoting = None, index = False)
    
    
    # Perform the training...
    model.predict (using_official_test = True, callback = callback)



if __name__ == "__main__":
    main ()
    
    
    