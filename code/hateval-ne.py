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

from numpy import unravel_index
from tqdm import tqdm
from dlsmodels.ModelResolver import ModelResolver
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


def main ():

    # var parser
    parser = DefaultParser (description = 'Hateval negation for task migrants and misogynistic', defaults = {
        'dataset': 'hateval'
    })
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()

    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var corpus String
    corpus = 'spain-2019'
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    original_dataset = dataset_resolver.get (args.dataset, corpus, '', False)
    original_dataset.filename = original_dataset.get_working_dir ('dataset.csv')


    # @var df Large Dataframe, contains satiric and non-satiric utterances
    df = original_dataset.get ()
    
    
    for target in ['mis', 'mig']:

        # @var dataset Dataset Get the dataset of the target
        dataset = dataset_resolver.get (args.dataset, 'spain-2019-' + target, '', False)
        dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
        
        
        # @var df_indexes Create a index list of the IDs we need to filter the negation features
        df_indexes = df.loc[df['target'] == target].index
        
        
        # @var ne_df Dataframe the negation features
        ne_df_full = pd.read_csv (dataset.get_working_dir ('ne_full.csv'))
        
        
        # Sumsample the same features that are in the dataframe in the same order
        ne_df = ne_df_full.loc[df_indexes].reset_index (drop = True)
    
    
        # Store in the target dataset
        ne_df.to_csv (dataset.get_working_dir ('ne.csv'), index = False)
    
        print (ne_df)

        
    
if __name__ == "__main__":
    main ()
