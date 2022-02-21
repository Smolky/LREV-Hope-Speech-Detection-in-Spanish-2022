"""
DatasetResolver

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import json
import sys
import os.path
import config
import importlib
from .Dataset import Dataset


class DatasetResolver ():
    """
    DatasetResolver
    
    This component returns the DatasetClass according to a key 
    identifiers (dataset, corpus, and task)
    
    """
    
    def get (self, dataset, corpus = '', task = '', refresh = False):
        """
        @param dataset string
        @param corpus string|null
        @param refresh string|null
        @param task boolean
        """
        
        # Security checkings
        if any ('/' in x for x in [dataset]):
            raise Exception ('Security checking contraint. Dataset cannot contain relative routes')
            sys.exit ()
        
        
        # @var config_path String Contains configuration details for this dataset
        config_path = os.path.join (config.directories['assets'], dataset, 'config.json')
        
        
        # If the configuration does not exists, we were looking for it in the 
        # general config folder. This is keep for compatibility
        if not os.path.isfile (config_path):
            config_path = '../config/dataset.json'
        
    
        # Load configuration of the dataset
        with open (config_path) as json_file:
            
            # @var dataset_options Dict
            dataset_options = json.load (json_file)[dataset]
            
            
            # If corpus is not supplied, then get the first element of our dataset
            corpus = list (dataset_options.keys())[0] if not corpus else corpus
            
            
            # @var corpus_options Retrieve data from configuration
            corpus_options = dataset_options[corpus]
        
        
        # @var args Dict 
        args = {
            'dataset': dataset, 
            'options': corpus_options, 
            'corpus': corpus, 
            'task': task, 
            'refresh': refresh
        }
        
        
        # Default implementation
        if not 'datasetClass' in corpus_options:
            return Dataset (**args)
        
        
        # @var module Import module dinamically
        module = importlib.import_module ("." + corpus_options['datasetClass'], package = 'dlsdatasets')
        
        
        # @var cls Loading the class dinamically
        cls = getattr (module, corpus_options['datasetClass'])
        
        return cls (**args)
    