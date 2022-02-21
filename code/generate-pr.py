"""
    To do random stuff
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle

from dlsmodels.ModelResolver import ModelResolver
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.BertEmbeddingsTransformer import BertEmbeddingsTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'To do random stuff')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()

    
    # var parser
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family or algorithms to evaluate', 
        choices = model_resolver.get_choices ()
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
    

    # @var features List
    features = ['lf', 'se', 'be', 'bf']
    
    
    # @var labels List
    labels = dataset.get_available_labels ()
    
    
    # @var dfs List
    dfs = []
    
    
    # Iterate over features
    for feature in features:
        
        # @var labels List
        new_labels = [feature + "_" + label for label in labels]


        # @var dfs_per_split List
        dfs_per_split = []
        
        
        # Over splits
        for split in ['train', 'val', 'test']:
    
            # @var probabilities_path String
            probabilities_path = dataset.get_working_dir (args.task, 'results', split, args.model, feature, 'probabilities.csv')
        
        
            # @var probabilities_df DataFrame
            probabilities_df = pd.read_csv (probabilities_path)
            probabilities_df = probabilities_df.rename (dict (zip (labels, new_labels)), axis = 1)
        
            dfs_per_split.append (probabilities_df[probabilities_df.columns.difference (['features', 'label'])])
        
    
        dfs.append (pd.concat (dfs_per_split, axis = 0))

    
    # @var predictions_df
    predictions_df = pd.concat (dfs, axis = 1).reset_index (drop = True)
    
    print (df)
    print (predictions_df)
    predictions_df.to_csv (dataset.get_working_dir (args.task, 'pr.csv'), index = False)
    
if __name__ == "__main__":
    main ()
