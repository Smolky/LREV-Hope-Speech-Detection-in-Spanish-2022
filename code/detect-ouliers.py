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
from scipy import stats

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver


def main ():

    # var parser
    parser = DefaultParser (description = 'Train dataset')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, args.force)
    
    
    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var train_df DataFrame Get training split
    train_df = dataset.get_split (df, 'train')
    
    
    # @var indexes Dict the the indexes for each split
    indexes = {split: subset.index for split, subset in {'train': train_df}.items ()}
    
    
    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)
    
    
    # @var features_cache String Retrieve 
    features_cache = dataset.get_working_dir ('lf.csv')
    
        
    # @var transformer
    transformer = feature_resolver.get ('lf', features_cache)
    
    
    # @var features_df DataFrame
    features_df = transformer.transform ([]);
    
    
    features_df = features_df[features_df.index.isin (indexes['train'])].reindex (indexes['train'])

    Q1 = df.quantile (0.25)
    Q3 = df.quantile (0.75)
    IQR = Q3 - Q1

    print (features_df)
    features_df = features_df[~((features_df < (Q1 - 1.5 * IQR)) | (features_df > (Q3 + 1.5 * IQR))).any (axis=1)]
    print (features_df)



if __name__ == "__main__":
    main ()
    
    
    