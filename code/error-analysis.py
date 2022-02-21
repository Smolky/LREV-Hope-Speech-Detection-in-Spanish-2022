"""
Error Analysis

@link https://neptune.ai/blog/deep-dive-into-error-analysis-and-model-debugging-in-machine-learning-and-deep-learning

@author José Antonio García-Díaz <joseantonio.garcia8@um.es>
@author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path
import pandas as pd

from pathlib import Path

from dlsdatasets.DatasetResolver import DatasetResolver
from dlsmodels.ModelResolver import ModelResolver
from dlsmodels.BaseModel import BaseModel
from features.FeatureResolver import FeatureResolver
from utils.Parser import DefaultParser


def main ():
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()
    
    
    # var parser
    parser = DefaultParser (description = 'Performs an error analysis')
    parser.add_argument ('--folder-base', dest = 'folder_baseline', help = 'Select the folder of the baseline model')
    parser.add_argument ('--folder', dest = 'folder', help = 'Select the folder of the model')
    
    
    # Add source
    parser.add_argument ('--source', 
        dest = 'source', 
        default = 'test', 
        help = 'Determines the source to make the error analysis', 
        choices = ['all', 'train', 'test', 'val']
    )
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var baseline_resume_file String
    baseline_resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder_baseline, 'training_resume.json')
    
    
    # ...
    baseline_resume = BaseModel.retrieve_training_info (baseline_resume_file)
    
    
    # @var baseline_model Model
    baseline_model = model_resolver.get_from_resume (baseline_resume)
    baseline_model.set_folder (args.folder)
    baseline_model.set_dataset (dataset)
    baseline_model.is_merged (dataset.is_merged)


    # @var baseline_resume_file String
    resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder, 'training_resume.json')


    # ...
    resume = BaseModel.retrieve_training_info (resume_file)


    # @var model Model
    model = model_resolver.get_from_resume (resume)
    model.set_folder (args.folder)
    model.set_dataset (dataset)
    model.is_merged (dataset.is_merged)
    
    
    # @var task_type String
    task_type = dataset.get_task_type ()


    # @var feature_resolver FeatureResolver
    feature_resolver = FeatureResolver (dataset)

    
    
    # Replace the dataset to contain only the test or val-set
    if args.source in ['train', 'val', 'test']:
        dataset.default_split = args.source
        

    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var y_real Series Get  real labels
    # @todo Update according to the task
    y_real = df['label']
    
    
    # @var labels List
    labels = dataset.get_available_labels ()
    
    
    # @var y_real_labels_available boolean
    y_real_labels_available = not pd.isnull (y_real).all ()
    if 'regression' == task_type:
        y_real_labels_available = None

    
    # @var predictions Dict
    predictions = {
        'probabilities_baseline': [],
        'baseline': [],
        'probabilities_model': [],
        'model': []
    }
    
    
    def callback (feature_key, y_pred, model_metadata):
        predictions['probabilities_model'] = model_metadata['probabilities'] if 'probabilities' in model_metadata else []
        predictions['model'] = y_pred.copy()
    
    
    # @var feature_combinations List
    feature_combinations = resume['features'] if 'features' in resume else {}


    # Load all the available features
    for feature_set, features_cache in feature_combinations.items ():
        
        # Indicate what features are loaded
        if not Path (features_cache).is_file ():
            continue
        
        
        # Set features
        model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))
            
    
    # Predict this feature set
    model.predict (using_official_test = True, callback = callback)
    model.clear_session ()
    

    
    def callback_baseline (feature_key, y_pred, model_metadata):
        predictions['probabilities_baseline'] = model_metadata['probabilities'] if 'probabilities' in model_metadata else []
        predictions['baseline'] = y_pred.copy()
    
    
    # @var feature_combinations List
    feature_combinations = baseline_resume['features'] if 'features' in resume else {}


    # Load all the available features
    for feature_set, features_cache in feature_combinations.items ():
        if not Path (features_cache).is_file ():
            print ("skip...")
            continue
        
        
        # Set features
        baseline_model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))


    # Predict this feature set
    baseline_model.predict (using_official_test = True, callback = callback_baseline)

    
    # Clear session
    baseline_model.clear_session ();
    
    
    # I want those models that are correctly classified by baseline model but not our best model
    for label_baseline, y_pred, label in zip (predictions['baseline'], predictions['model'], y_real.tolist ()):
        if label_baseline == label and y_pred != label:
            print ("bingo!")
    
    
    print ()
    
    # ...
    index = 0
    data = []
    for probab, y_pred, label in zip (predictions['probabilities_model'], predictions['model'], y_real.tolist ()):
        if y_pred != label and label != 'neutral':
            label_index = [labels.index (label)]
            if label_index == 0:
                label_index = 2
            elif label_index == 2:
                label_index = 0
            
            if probab[label_index] >= .45:
                data.append ({
                    'tweet': df.iloc[index]['tweet_clean'],
                    'label': label,
                    'probab': probab,
                })
        index = index + 1

    print (pd.DataFrame.from_records (data).to_latex ())

if __name__ == "__main__":
    main ()