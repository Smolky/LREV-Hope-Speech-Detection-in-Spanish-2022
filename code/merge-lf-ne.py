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

from tqdm import tqdm
from dlsmodels.ModelResolver import ModelResolver
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver

from features.LinguisticFeaturesTransformer import LinguisticFeaturesTransformer
from features.SentenceEmbeddingsTransformer import SentenceEmbeddingsTransformer
from features.BertEmbeddingsTransformer import BertEmbeddingsTransformer
from features.TokenizerTransformer import TokenizerTransformer



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


    # @var language String
    language = dataset.get_dataset_language ()


    # @var modes List
    modes = [
        '_robust_ig',
        '_robust_anova',
        '_robust',
        '_minmax_ig',
        '_minmax_anova',
        '_minmax',
        '_ig'
    ]
    
    
    # @var modes List
    features = ['lf', 'ne']
    

    # Iterate modes
    for mode in modes:
        
        print ()
        print (mode)
        
        # @var temp Dict
        temp = []
        
        for feature in tqdm (features):
            
            # @var features_cache String The file where the features are stored
            features_cache = dataset.get_working_dir (args.task, feature + mode + '.csv')
        
            
            # Load features
            temp.append (pd.read_csv (features_cache))
            

        # @var merged_df DataFrame
        merged_df = pd.concat (temp, axis = 1)
        
        
        # @var file String
        file = dataset.get_working_dir (args.task, features[0] + mode + '.csv')
        
        
        # Store
        merged_df.to_csv (file, index = False)
    
if __name__ == "__main__":
    main ()
