"""
    iSarcasm put the labels in the test dataset
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import csv
import json
import sklearn
import itertools
import re

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


def main ():

    # @var dataset_name String
    dataset_name = 'isarcasm'
    
    
    # var parser
    parser = DefaultParser (description = 'iSarcasm Test Labels')


    # @var args Get arguments
    args = parser.parse_args ()
    

    # @var corpus_name String
    for corpus_name in ['2022-en', '2022-ar']:
        
        # Retrieve tasks
        if '2022-en' == corpus_name:
            tasks = ['task-a', 'task-b-irony', 'task-b-overstatement', 'task-b-rhetorical_question', 'task-b-sarcasm', 'task-b-satire', 'task-b-understatement']
            language = 'En'
        
        if '2022-ar' == corpus_name:
            tasks = ['task-a']
            language = 'Ar'
        
        
        
        # @var task_name String
        for task_name in tasks:
            
            # @var task_letter String
            task_letter = 'A' if '-a' in task_name else 'B'
            
            
            # @var labels_file String
            labels_file = 'task_' + task_letter + '_' + language + '_test.csv'
            
            
            # Specify the rest of the args
            args.dataset = dataset_name
            args.corpus = corpus_name
            args.task = task_name
        
        
            # @var dataset_resolver DatasetResolver
            dataset_resolver = DatasetResolver ()
            
            
            # @var dataset Dataset This is the custom dataset for evaluation purposes
            dataset = dataset_resolver.get (dataset_name, corpus_name, task_name, False)

            # @var task_letter String
            label_field = 'sarcastic' if task_letter == 'A' else dataset.options['tasks'][task_name]['isarcasm_label']
            
            
            # @var df_labels DataFrame
            df_labels = pd.read_csv (dataset.get_working_dir ('dataset', labels_file))

            
            # Determine if we need to use the merged dataset or not
            dataset.filename = dataset.get_working_dir (task_name, 'dataset.csv')
            
            
            # @var df DataFrame
            df = dataset.get ()
            
            
            # Assign the label
            df.loc[df['__split'] == 'test', 'label'] = df_labels[label_field].tolist ()
            
            
            # Change name to label
            if task_letter == 'A':
                df.loc[df['label'] == 1, 'label'] = "sarcasm"
                df.loc[df['label'] == 0, 'label'] = "non-sarcasm"
            else:
                df.loc[df['label'] == 1, 'label'] = label_field
                df.loc[df['label'] == 0, 'label'] = "non-" + label_field
            
            
            # Save
            df.to_csv (dataset.filename, index = False, quoting = csv.QUOTE_ALL)


if __name__ == "__main__":
    main ()
