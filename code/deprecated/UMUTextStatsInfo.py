"""
    Obtain LF Info
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import requests
import sys
import csv
import itertools
import subprocess
import os.path
import config
import argparse
import io
import numpy as np
import pandas as pd

from io import StringIO
from pathlib import Path
from datasetResolver import DatasetResolver
from UMUTextStatsTransformer import UMUTextStatsTransformer
from Tagger import Tagger


def main ():
    """ To use from command line """

    # Parser
    parser = argparse.ArgumentParser (description='Obtain LF information')
    parser.add_argument ('--dataset', dest = 'dataset', default = next (iter (config.datasets)), help="|".join (config.datasets.keys ()))
    parser.add_argument ('--force', dest = 'force', default = False, help = "If True, it forces to replace existing files")
    parser.add_argument ('--merge', dest = 'merge', default = False, help = "If True, it merges all the tweets per author")
    parser.add_argument ('--task', dest = 'task', default = "", help = "Optional, task")


    # Get args
    args = parser.parse_args ()
    
    
    # Iterate over the subdatasets
    for key, dataset_options in config.datasets[args.dataset].items ():

        # Get the dataset name
        dataset_name = args.dataset + "-" + key + ".csv"
        
        
        # Resolver
        resolver = DatasetResolver ()
        
        
        # @var tagger Tagger 
        tagger = Tagger (dataset_options['language'])
        
        
        # Get the dataset name
        dataset_name = args.dataset + "-" + key + '.csv'
        
        
        # Get the dataset
        dataset = resolver.get (dataset_name, dataset_options, args.force)
        
        
        # Get the dataset as a dataframe
        df = dataset.get (args.task)
        
        
        # For testing purposes. 
        # This code will focus on the specific ids you want to look into
        # Morever, to correctly display the progress bar, you need to reset the index
        """
        df = df.loc[1:500, :]
        df = df.reset_index ()
        """
        
        
        # Tags
        tags = tagger.get (df)
        
        
        # Update dataframe
        df = df.assign (tagged_pos = tags['pos'])
        df = df.assign (tagged_ner = tags['ner'])
        
        
        # LF Transformer
        lf = UMUTextStatsTransformer ()
        
        
        # Get features
        results = lf.transform (df)
        
        
        # Merge features
        results = results.assign (label = df['user'])
                
        
        print ()
        results = results.groupby (['label']).mean ().reset_index ()
        
        print (results)
        sys.exit ()
        
        # Describe
        pd.set_option ('display.max_rows', results.shape[1] + 1)
        print ()
        print (results.describe ().T)
        
        
        
        
        sys.exit ()
    
    
if __name__ == "__main__":
    main ()