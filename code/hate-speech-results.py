"""
    Hate-Speech paper
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle
import io

from contextlib import redirect_stdout
from pathlib import Path
from sklearn.metrics import classification_report

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser


def main ():


    # var parser
    parser = DefaultParser (description = 'Generate latex tables')
    
    
    # @var datasets List
    datasets = [
        {"dataset": "misogyny", "corpus": "misocorpus", "name": "Spanish MisoCorpus 2020"},
        {"dataset": "ami", "corpus": "2018", "name": "AMI 2018"},
        {"dataset": "haternet", "corpus": "2019", "name": "HaterNET"},
        {"dataset": "hateval", "corpus": "spain-2019", "name": "HatEval 2019"}
    ]
    
    
    # @var experiments Dict
    experiments = {

        'ensemble': {
            'caption': 'Ensemble',
            'label': 'tab:results-ensembles',
            'model': 'ensemble',
            'feature_combinations': [['mode'], ['weighted mode'], ['h. probab'], ['mean']]
        }
    }
    """
    'isolation': {
        'feature_combinations': [['lf'], ['se'], ['we'], ['bf']],
        'caption': 'Classification report for individual features',
        'label': 'tab:results-isolation',
        'model': 'deep-learning',
        'n': 4,
    },
    'integration': {
        'feature_combinations': [['lf', 'bf'], ['se', 'we', 'bf'], ['lf', 'se', 'we', 'bf']],
        'caption': 'Knowledge integration',
        'label': 'tab:results-integration',
        'model': 'deep-learning'
    },"""    
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()


    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    for experiment, experiment_stuff in experiments.items ():

        print (experiment)
        print ("----------------------------------")

        # @var experiment_formated_results Dict
        experiment_formated_results = {}
        experiment_formated_results[experiment] = {}


        # @var experiment_results Dict
        experiment_results = {}
        experiment_results[experiment] = {}

    
        # Iterate over the results
        for dataset_info in datasets:

            print (dataset_info['dataset'])

            # @var dataset_results List
            dataset_results = {}


            # @var dataset_formated_results List
            dataset_formated_results = {}

        
            def callback (feature_key, y_pred, model_metadata):
                """
                @var feature_key String
                @var y_pred Series
                @var model_metadata Dict
                """
                
                # @var report DataFrame|None
                report = pd.DataFrame (classification_report (
                    y_true = y_real, 
                    y_pred = y_pred, 
                    digits = 5,
                    output_dict = True
                )).T
                report = report.drop (['support'], axis = 1)
                # report = report.head (2)
                report = report.drop (['accuracy', 'macro avg'], axis = 0)
            
            
                # Attach data
                dataset_results[feature_key.upper ()] = report
                
                
                # Transform in percentage
                for column in report.columns:
                    report[column] = report[column].apply (lambda x: x * 100)
                
                
                # Render as string to print all decimals
                report = report.astype (str)


                # With \\no, we can decide the number of decimals later
                for column in report.columns:
                    report[column] = report[column].apply (lambda x: "\\no{" + x + "}")
                
                # Attach data
                dataset_formated_results[feature_key.upper ()] = report

            
            # @var dataset Dataset This is the custom dataset for evaluation purposes
            dataset = dataset_resolver.get (dataset_info['dataset'], dataset_info['corpus'], '', False)
            dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


            # @var model ModelInterface
            model = model_resolver.get (experiment_stuff['model'])
            model.set_dataset (dataset)
            dataset.default_split = 'test'


            # @var y_real Series
            y_real = dataset.get ()['label']
            
            
            # @var task_type String
            task_type = dataset.get_task_type ()
            
            
            # @var feature_resolver FeatureResolver
            feature_resolver = FeatureResolver (dataset)


            # Load all the available feature combinations
            if model.has_external_features ():
            
                print ("loading external features")
            
                for features in experiment_stuff['feature_combinations']:
                
                    # Load the features
                    for feature_set in features:
                
                        # @var feature_file String
                        feature_file = feature_resolver.get_suggested_cache_file (feature_set, task_type)

                    
                        # @var features_cache String The file where the features are stored
                        features_cache = dataset.get_working_dir (args.task, feature_file)

                        
                        # If the feautures are not found, get the default one
                        if not Path (features_cache, cache_file = "").is_file ():
                            features_cache = dataset.get_working_dir (args.task, feature_set + '.csv')
        
                    
                        # Set features
                        model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))    
                
                
                    # Predict this feature set
                    model.predict (using_official_test = True, callback = callback)
        
                    
                    # Clear session
                    model.clear_session ();

            else:

                print ("features will be loaded within the ensemble")


                # Predict this feature set
                model.predict (using_official_test = True, callback = callback)


                # Clear session
                model.clear_session ();


            # @var merged_formated_df DataFrame
            merged_formated_df = pd.concat (dataset_formated_results.values (), axis = 1, keys = dataset_formated_results.keys ())
            merged_df = pd.concat (dataset_results.values (), axis = 1, keys = dataset_results.keys ())
            
            """
            print (merged_formated_df)
            print (merged_df)
            print (merged_df.max (axis = 1))
            sys.exit ()
            """

            # Attach data
            experiment_formated_results[experiment][dataset_info['name']] = merged_formated_df
            experiment_results[experiment][dataset_info['name']] = merged_df

        
        # @var num_columns int
        num_columns = 1 + (3 * len (experiment_stuff['feature_combinations']))
        

        # @var column_format String
        column_format = 'l';
        for i in range (len (experiment_stuff['feature_combinations'])):
            column_format += 'ccc|';
        column_format = column_format.rstrip ('|')
        
        
        # @var features_str String
        features_str = "             & "
        metrics_str = "{}           & "
        
        for feature in experiment_stuff['feature_combinations']:
            features_str += "\\multicolumn{3}{c}{" + '+'.join (feature) + "} & "
            metrics_str += "P            & R            & F1           & "
            
        features_str = features_str.rstrip ('& ')
        features_str = features_str + "\\\\"
        metrics_str = metrics_str.rstrip ('& ')
        metrics_str = metrics_str + "\\\\"
        
        
        
        # Iterate to get the final result
        with open ('outputs/' + experiment + '.latex', 'w') as f:
            with redirect_stdout(f):
                print ("\\begin{table}")
                print ("\\centering")
                print ("\\caption{" + experiment_stuff['caption'] + "}\\label{tab:" + experiment_stuff['label'] + "}")
                print ("\\npdecimalsign{.}")
                print ("\\nprounddigits{2}")
                print ("\\begin{tabular}{" + column_format + "}")
                print ("\\toprule")

                for dataset, table in experiment_formated_results[experiment].items ():
                    print ("\\multicolumn{" + str (num_columns) + "}{c}{" + dataset + "} \\\\")
                    print ("\\midrule")
                    print (features_str)
                    print ("\\midrule")
                    print (metrics_str)
                    print ("\\midrule")
                    table_latex = table.to_latex (column_format = column_format, header = False, escape = False)
                    for line in io.StringIO (table_latex):
                        if not line.startswith ('\\'):
                            print (line.replace ('-', '\\-').replace ('_', '\\_'))
                    
                    if dataset != 'HatEval 2019':
                        print ("\\midrule")
                
                print ("\\bottomrule")
                print ("\end{tabular}")
                print ("\\end{table}")
    
if __name__ == "__main__":
    main ()