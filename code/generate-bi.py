"""
    Generate BEIT embeddings
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.BeitEmbeddingsTransformer import BeitEmbeddingsTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Generate BEIT embeddings from the finetuned model')
    
    
    # Add parser
    parser.add_argument ('--folder', dest = 'folder', default = '', help = 'Select a folder for the model')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()
    
    
    # @var cache_file String
    cache_file = dataset.get_working_dir (dataset.task, 'bi.csv')
    
    
    # @var pretrained_model String
    pretrained_model = 'microsoft/beit-base-patch16-224-pt22k-ft22k'
    
    
    # @var be_transformers BertEmbeddingsTransformer
    bi_transformers = BeitEmbeddingsTransformer (
        pretrained_model, 
        cache_file = cache_file, 
        image_path = dataset.get_working_dir ('images'),
        field = 'twitter_id'
    )

    print (bi_transformers.transform (df))

    
if __name__ == "__main__":
    main ()
