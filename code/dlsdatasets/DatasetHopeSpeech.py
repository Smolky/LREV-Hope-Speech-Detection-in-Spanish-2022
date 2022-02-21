import csv
import sys
import string
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from .Dataset import Dataset
from scipy import stats


class DatasetHopeSpeech (Dataset):
    """
    DatasetMadrid
    

    @extends Dataset
    """

    def __init__ (self, dataset, options, corpus = '', task = '', refresh = False):
        """
        @inherit
        """
        Dataset.__init__ (self, dataset, options, corpus, task, refresh)
    
    def compile (self):
        
        # @var df Load dataframes
        df = pd.read_csv (self.get_working_dir ('dataset', 'dataset.csv'))
        
        
        # Strip blank lines
        df = df.replace (to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex = True)
        
        
        # Change class names
        df = df.rename (columns = {
            'id': 'twitter_id',
            'text': 'tweet',
            'category': 'label'
        })
        
        
        # Generate the splits
        df = self.assign_default_splits (df)
        
        
        
        # Store this data on disk
        self.save_on_disk (df)
        
        
        # Return
        return df
        