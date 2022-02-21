"""
    Generate folds
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

def main ():

    # var parser
    parser = DefaultParser (description = 'Generate folds')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()
    
    
    df = dataset.assign_default_folds (df)
    
    print (df)
    print (df['__split_fold_1'].value_counts ())
    
    # dataset.save_on_disk (df)
    
    
if __name__ == "__main__":
    main ()
