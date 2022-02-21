"""
    HASOC put test labels
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import csv
import config
import bootstrap
import os.path
import pandas as pd

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


def main ():
    
    # var parser
    parser = DefaultParser (description = 'HASOC labels')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get ('hasoc', args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # Load labels
    labels_df = pd.read_csv (dataset.get_working_dir ('dataset', 'test_labels_' + args.task + '.csv'))
    labels_df = labels_df.set_index ('id')
    
    
    # @var df Dataframe
    df = dataset.get ()
    df = df.set_index ('twitter_id', drop = False)


    # @var label_field String
    label_field = dataset.options['tasks'][args.task]['label']
    
    print (label_field)
    
    
    # Assign labels
    df.loc[df['__split'] == 'test', label_field] = labels_df['label']
    df = df.reset_index (drop = True)

    print (df.loc[df['__split'] == 'test'][label_field])


    # Refresh
    dataset.save_on_disk (df)
    


if __name__ == "__main__":
    main ()