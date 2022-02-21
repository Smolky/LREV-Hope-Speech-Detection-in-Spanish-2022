"""
    To do random PAN stuff
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion


def main ():

    # var parser
    parser = DefaultParser (description = 'To evaluate PAN')
    
    
    # Attach extra stuff
    parser.add_argument ('--merged', dest = 'merged', default = False, help = 'In author profiling, get the merged dataset')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver () 
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir ('merged' if args.merged else '', 'dataset.csv') 
    dataset.is_merged = args.merged


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    # @for testing
    # df = df.iloc[0:200, :]
    # df = df[df['user'] == "5f55d9af16dc96e60450e82d5b8a1a12"]
    # fef2ba0032528112febdd0ba49fe23b7
    # 799f8e04f3f35cd27f59d34906fe3dd
    # df = df.reset_index ()
    # print (df[df['tweet_clean'].str.contains("Muy simpático")]['tweet_clean']) 
    # sys.exit ()
    
    print (df.loc[167901])
    print (df.loc[340104])
    print (df.loc[167900])
    
    df = df.reset_index ()
    
    """
    print ()
    print (df['tweet_clean'].iloc[0])
    print (df['user'])
    print (df['label'])
    """
    
    sys.exit ()


if __name__ == "__main__":
    main ()
    
    
    