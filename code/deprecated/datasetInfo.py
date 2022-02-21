"""
    DatasetInfo
    
    Obtains information regarding the dataset
    
    @see config.py
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import config
import sys
import argparse
import os.path
import csv
import pandas as pd

from dataset import Dataset
from datasetResolver import DatasetResolver


       

def main ():
    """ To use from command line """
    
    # Parser
    parser = argparse.ArgumentParser (description = 'Obtains information regarding the dataset')
    parser.add_argument ('--dataset', dest = 'dataset', default=next (iter (config.datasets)), help="|".join (config.datasets.keys ()))
    parser.add_argument ('--task', dest = 'task', default = '', help = 'Get the task')
    

    # Get args
    args = parser.parse_args ()
    
    
    # datasets
    datasets = config.datasets[args.dataset].items ()
    
    
    # @var umucorpus_ids int|string The Corpus IDs
    for key, dataset_options in datasets:
        
        # dataset_name
        dataset_name = args.dataset + "-" + key + ".csv"
        
        
        # @var resolver DatasetResolver
        resolver = DatasetResolver ()
        
        
        # @var dataset Get the most suitable dataset
        dataset = resolver.get (dataset_name, dataset_options, False)
        
        
        # @var df DataFrame Get the dataset as a dataframe
        df = dataset.get (args.task)
        
        
        # Get datasets splits for training, validation and testing
        train_df = dataset.get_split (df, 0)
        val_df = dataset.get_split (df, 1)
        test_df = dataset.get_split (df, 2)
        
        
        # Determine if the same user appears in different splits
        print ("duplicated users")
        print ("----------------")
        users = [
            set (train_df['user'].unique ()),
            set (val_df['user'].unique ()),
            set (test_df['user'].unique ())
        ]
        
        print (set.intersection (*users))
        
        
        # 
        for field in ['nature', 'gender', 'age_range', 'ideological_binary', 'ideological_multiclass']:
        
            if field in df.columns:
        
                print ()
                print (field)
                print ("=============================")
            
                df_resume = pd.DataFrame.from_dict ({
                    'full': df[field].value_counts (),
                    'train': train_df[field].value_counts (),
                    'val': val_df[field].value_counts (),
                    'test': test_df[field].value_counts ()
                }, orient = 'columns')
                
                print (df_resume)
        

if __name__ == "__main__":
    main ()