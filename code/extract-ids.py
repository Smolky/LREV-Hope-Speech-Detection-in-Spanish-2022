"""
    Show dataset statistics for label and split distribution
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path
import pandas as pd

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Compile dataset')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # Keep train, validation, and test
    df = df.loc[~df['__split'].isna ()]
    
    
    df_ids = df[['twitter_id', 'label', '__split']]
    df_ids.to_csv (dataset.get_working_dir ('dataset_ids.csv'), index = False)


    labels = ['twitter_id', 'label', '__split', 'user', 'tweet'] if 'user' in df.columns else ['twitter_id', 'label', '__split', 'tweet']
    
    
    df_ids = df[labels]
    df_ids.to_csv (dataset.get_working_dir ('dataset_ids_texts.csv'), index = False)
    

if __name__ == "__main__":
    main ()