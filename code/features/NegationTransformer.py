import pandas as pd
import pickle
import os.path

from tensorflow import keras
from keras import backend as K
from pathlib import Path
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 

class NegationTransformer (BaseEstimator, TransformerMixin):
    """
    Obtain sentences tokenized

    """

    def __init__ (self, cache_file = ''):
        """
        @param model String (see Config)
        @param cache_file String
        """
        super().__init__()
        
        self.cache_file = cache_file
        self.columns = None
        
    
    # Return self nothing else to do here
    def fit (self, X, y = None ):
        return self 
        
    def transform (self, X, **transform_params):
    
        # Return tokens from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")

        print (self.cache_file)