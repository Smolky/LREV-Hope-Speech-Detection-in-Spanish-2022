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
from utils.LabelsDistribution import LabelsDistribution
from utils.WordsCloud import WordsCloud
from utils.CorpusStatistics import CorpusStatistics



def main ():
    
    # var parser
    parser = DefaultParser (description = 'Show statistics from the datasets')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var split_field String
    split_field = '__split'
    
    
    # @var label_field String
    label_field = 'label'
    
    
    # @var grouping_per_user Boolean Determines if the statistics of the dataset 
    #                       should be grouped by tweet (default) or by 
    #                       group
    grouping_per_user = dataset.group_results_by_user ()
    
    
    # Keep train, validation, and test
    df = df.loc[~df[split_field].isna ()]
    df = df.loc[df[split_field].isin (['train', 'val', 'test'])]
    
    
    # @var corpus_statistics CorpusStatistics
    corpus_statistics = CorpusStatistics (dataset)
    
    
    with pd.option_context('display.max_rows', 1000, 'display.max_columns', 10):
        if 'twitter_created_at' in df.columns:
            print ()
            print ("dataset dates")
            print ("--------------------")
            
            df_dates = df
            df_dates['twitter_created_at'] = pd.to_datetime (df_dates['twitter_created_at'], errors = 'coerce')
            
            print ()
            print ("min and max dates")
            print (df_dates.set_index ('twitter_created_at').index.min ())
            print (df_dates.set_index ('twitter_created_at').index.max ())
            
            print ()
            print ("distribution")
            print (df_dates['twitter_created_at'].groupby (df_dates['twitter_created_at'].dt.to_period ("M")).agg ('count'))



        print ()
        print ("dataset distribution")
        print ("--------------------")
        print (df[split_field].value_counts (sort = False).sort_index ())
        
        
        if 'user' in df.columns:
            print ()
            print ("user distribution")
            print ("-----------------")
            
            # @var user_counts
            user_counts = df['user'].value_counts (sort = False)
            
            
            # @var how_many_different_users int
            how_many_different_users = len (user_counts)
            
            
            if how_many_different_users >= 10:
                print ("The mean is {}".format (user_counts.mean ()))
                print ("The median is {}".format (user_counts.median ()))
                print ("The mode is {}".format (user_counts.mode ()))
                print ("The standard deviation is {}".format (user_counts.std ()))
            
            else:
                print (user_counts.sort_index ())
                print (df['user'].value_counts (normalize = True, sort = False).sort_index ())
        

        if dataset.get_task_type () == 'classification' and not grouping_per_user:
        
            print ()
            print ("label distribution")
            print ("------------------")
            print (df[label_field].value_counts (sort = False).sort_index ())
            for split in ['train', 'val', 'test']:
                print ()
                print ("label distribution in the " + split + " split")
                print (dataset.get_split (df, split, split_field = split_field)[label_field].value_counts (sort = False).sort_index ())
        
        elif dataset.get_task_type () == 'multi_label':
            
            df_labels = df[label_field].str.split ('; ', expand=True)
            columns = [df_labels[column] for column in df_labels]
            df_labels = pd.concat (columns, axis = 0, ignore_index = True).dropna (axis='rows')
            print (df_labels.value_counts (sort = False).sort_index ())
        
        elif dataset.get_task_type () == 'regression':
            print (df.loc[df[label_field].notnull()][label_field].describe ())
            print (df.loc[df[label_field].notnull()][label_field].mode ())
            
            sys.exit ()
    
    
    print ()
    print ("columns distribution")
    print (corpus_statistics.get_columns_distribution_in_different_splits ())
    
    print ()
    print ("line length distribution")
    print (corpus_statistics.get_line_length_distribution ())
    
    print ()
    print ("duplicated labels")
    print (corpus_statistics.get_duplicated_labels_in_different_splits (label = label_field))

if __name__ == "__main__":
    main ()