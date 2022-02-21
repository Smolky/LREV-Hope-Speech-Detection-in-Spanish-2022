import sys
import os.path
import io
import csv
import math
import bootstrap
import numpy as np
import pandas as pd
import sklearn
import joblib
import pickle
import time

from pathlib import Path
from pipelinehelper import PipelineHelper

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report

import dask_ml.model_selection as dcv
from dask.diagnostics import ProgressBar

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest


from .BaseModel import BaseModel


class TFIDFModel (BaseModel):
    """
    TFIDFModel
    
    This class allows to train and use different word and n-grams. This model 
    is usually employed as baseline
    
    @todo Adapt to regression
    @todo Adapt to multiclassfication
    @todo Parametrize models and architecture
    """
    
    # @var all_available_features List
    all_available_features = []
    
    
    @classmethod
    def adapt_argument_parser (cls, parser):
        return False

    def set_arguments_parser (self, args):
        """
        @param args
        """
        return False
    
    def has_external_features (self):
        """
        @inherit
        """        
        return False
        
        
    def get_folder (self):
        """
        @inherit
        """
        return self.folder or 'tf-idf'
        
        
    def get_classifiers (self):
        """
        The classifiers to be evaluated
        """
        return [
            # ('mnb_classifier', MultinomialNB ()),
            # ('lr', LogisticRegression (max_iter = 4000)),
            ('svm', SVC (probability = True)),
            # ('k_classifier', KNeighborsClassifier (n_neighbors = 2)),
            # ('j48', DecisionTreeClassifier ()),
            # ('rf', RandomForestClassifier (bootstrap = False, max_features = 'auto', min_samples_leaf = 2, min_samples_split = 2))
        ]
    
    def train (self, force = False, using_official_test = True):
        """
        @inherit
        """
        
        # @var dataset_options Dict
        dataset_options = self.dataset.get_task_options ()
        
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var train_df DataFrame Get training split. If using the validation set for training
        train_df = pd.concat ([
            self.dataset.get_split (df, 'train'), 
            self.dataset.get_split (df, 'val')]
        )
        
        
        # @var val_df DataFrame
        val_df = self.dataset.get_split (df, 'val')
        
        
        # @var model_folder String
        model_folder = self.get_folder ()
        
        
        # @var available_labels
        available_labels = self.dataset.get_available_labels ()
        
        
        # @var split_index List Train data indices are -1 and validation data indices are 0
        # @link https://stackoverflow.com/questions/31948879/using-explicit-predefined-validation-set-for-grid-search-with-sklearn
        split_index = [0 if x in val_df.index else -1 for x in train_df.index]

    
        # @var split int Create a Predefined split based on my indexes
        split = sklearn.model_selection.PredefinedSplit (test_fold = split_index)
        
        
        # @var classifiers Dict Classifiers to evaluate and their default parameters
        classifiers = self.get_classifiers ()
        
        
        # @var scoring_metric
        scoring_metric = self.dataset.get_scoring_metric ().replace ('val_', '')
        
        
        # Adjust F1-Score metric
        if 'scoring' in dataset_options:
            if dataset_options['scoring'] == 'micro':
                scoring_metric = 'f1_micro'
            
            elif dataset_options['scoring'] == 'macro':
                scoring_metric = 'f1_macro'

            elif dataset_options['scoring'] == 'weighted':
                scoring_metric = 'f1_weighted'
            
            else:
                scoring_metric = 'f1_weighted'
                
        elif scoring_metric == 'f1_score':
            scoring_metric = 'f1_weighted'


        # @var n_iter int Number of models to test
        n_iter = 5
        
        
        for classifier in classifiers:
        
            print (classifier[0])
            print ("---------------")
        
        
            # Analyzer
            for analyzer in ['word', 'char_wb']:
            
                print (analyzer)
                
                # @var features__ngram_ranges List
                features__ngram_ranges = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)] if analyzer == 'word' else [(4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10)]
            
            
                # Iterate over ranges...
                for features__ngram_range in features__ngram_ranges:
                
                    print (features__ngram_range)
                
                    # @var pipe Pipeline Create pipeline with the feature selection and the classifiers
                    # @todo Bring combination of the parameters in the pipeline
                    pipe = Pipeline ([
                        ('features', TfidfVectorizer ()),
                        ('select', PipelineHelper ([
                            ('vt', VarianceThreshold ()),
                            ('skbest', SelectKBest ()),
                        ])),
                        ('classifier', PipelineHelper ([classifier]))
                    ])
                
                
                    # @var hyperparameters Dict  
                    hyperparameters = {}
                
                
                    # This are the TFIDF vectorizer options
                    features_options = {
                        'features__sublinear_tf': [True, False],
                        'features__strip_accents': [None, 'unicode'],
                        'features__use_idf': [True, False],
                    }
                    

                    if analyzer == 'word' and features__ngram_range != (3, 3):
                        features_options['features__max_df'] = [0.01, 0.1, 1]

                    
                    if classifier[0] == 'rf':
                        # Define some parameters space beforehand
                        # @var rf__max_depth
                        rf__max_depth = [int(x) for x in np.linspace (10, 110, num = 5)]
                        rf__max_depth.append (None)
                    
                        
                        # @var rf__n_estimadors List
                        rf__n_estimadors = [int (x) for x in np.linspace (start = 200, stop = 2000, num = 5)]
                    
                
                        # @var hyperparameters Dict  
                        hyperparameters['rf__n_estimators'] = rf__n_estimadors
                        hyperparameters['rf__max_features'] =  ['auto', 'sqrt']
                        hyperparameters['rf__max_depth'] =  rf__max_depth
                        hyperparameters['rf__min_samples_split'] =  [2, 5, 10]
                        hyperparameters['rf__min_samples_leaf'] =  [1, 2, 4]

                    elif classifier[0] == 'lr':
                        hyperparameters['lr__solver'] = ['liblinear', 'lbfgs']
                        hyperparameters['lr__fit_intercept'] = [True, False]
                        
                    elif classifier[0] == 'svm':
                        hyperparameters['svm__C'] = [1]
                        hyperparameters['svm__kernel'] = ['rbf', 'poly', 'linear']
                    
                    
                    
                    # @var classifier_hyperparameters Filter only those parameters related to the classifiers we use
                    classifier_hyperparameters = {key: hyperparameter for key, hyperparameter in hyperparameters.items () 
                                                if key.startswith (tuple ([(classifier_key[0] + "__") for classifier_key in classifiers]))}
            

                    # @var parameters Dictionary
                    parameters = {
                        'classifier__selected_model': pipe.named_steps['classifier'].generate (classifier_hyperparameters)
                    }
                
            
                    # Create the specific bag of word features from unigrams to trigrams
                    features = {
                        'features__analyzer': [analyzer],
                        'features__ngram_range': [features__ngram_range]
                    }
                    
                    
                    # Mix the specific and generic parameters for the character n-grams and the word-grams
                    features = {**features, **features_options}
                    
                    
                    # Mix the features with the classifier parameters
                    features['classifier__selected_model'] = pipe.named_steps['classifier'].generate (classifier_hyperparameters)
                    
                    
                    # Parameters of pipelines can be set using ‘__’ separated parameter names:
                    param_grid = [features]
                    
                    
                    # @var best_model_file String
                    best_model_file = self.dataset.get_working_dir (self.dataset.task, 'models', 'tf-idf', classifier[0], analyzer, '_'.join (map (str, features__ngram_range)), 'best_model.joblib')
                    
                    
                    # If the file exists, then skip (unless we force to retrain)
                    if os.path.isfile (best_model_file):
                        if not force:
                            print ("...skip")
                            continue
            
            
                    # @var search RandomizedSearchCV
                    search = sklearn.model_selection.RandomizedSearchCV (pipe, param_grid, 
                        cv = split, 
                        n_iter = n_iter, 
                        scoring = scoring_metric, 
                        random_state = bootstrap.seed,
                        refit = True
                    )
                
                    
                    # Fit
                    with ProgressBar ():
                        search.fit (train_df['tweet_clean_lowercase'], train_df['label'].astype ('category').cat.codes)
                    
                    
                    # @var df_summary DataFrame
                    df_summary = pd.DataFrame (search.cv_results_) \
                        .drop (columns = ['std_fit_time', 'std_score_time', 'param_classifier__selected_model', 'std_test_score', 'split0_test_score'], axis = 1) \
                        .sort_values ('rank_test_score', ascending = True) \
                        [['rank_test_score', 'mean_test_score', 'mean_fit_time', 'mean_score_time', 'params']]
                    
                
                    # Refit the estimator
                    # @var link https://stackoverflow.com/questions/57059016/refit-attribute-of-grid-search-with-pre-defined-split
                    
                    # @todo @fixme
                    # This way the estimator ir refiting twice, as we 
                    # set refit = True. Maybe the idea is not to use 
                    # refit, retrieve the parameters from df_summary and 
                    # retrain this way
                    with ProgressBar ():
                        search.best_estimator_.fit (train_df['tweet_clean_lowercase'], train_df['label'].cat.codes)
                    
                    
                    # Store the results for further anaylsis
                    joblib.dump (search.best_estimator_, best_model_file)
                
                
                    # @var summary_path String
                    summary_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, classifier[0], analyzer, '_'.join (map (str, features__ngram_range)), 'hyperparameters.csv')
                    
                    
                    # Output summaries
                    df_summary.to_csv (summary_path, index = False, quoting = csv.QUOTE_ALL)

    
    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        
        @todo using_official_test
        
        """

        # @var classifiers Dict Classifiers to evaluate and their default parameters
        classifiers = self.get_classifiers ()

        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var true_labels
        true_labels = self.dataset.get_available_labels ()


        # Iterate by classifiers
        for classifier in classifiers:
        
            # Iterate by analyzers
            for analyzer in ['word', 'char_wb']:
            
                # @var features__ngram_ranges List
                features__ngram_ranges = [(1, 1), (1, 2), (2, 2), (1, 3), (2, 3), (3, 3)] if analyzer == 'word' else [(4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10)]
                
                
                # Iterate by ngram range
                for features__ngram_range in features__ngram_ranges:
        
                    # @var model_filename String
                    model_filename = self.dataset.get_working_dir (self.dataset.task, 'models', 'tf-idf', classifier[0], analyzer, '_'.join (map (str, features__ngram_range)), 'best_model.joblib')
                
                
                    # Ensure them model exists
                    if not os.path.exists (model_filename):
                        sys.exit ()
                        return

                    
                    # @var best_model
                    try:
                        best_model = joblib.load (model_filename)
                    
                    except:
                        print ("model could not be loaded. Skip...")
                        sys.exit ()
                        return

                    
                    # @var y_pred List
                    y_pred = [true_labels[int (item)] for item in best_model.predict (df['tweet_clean_lowercase'])]
                    
                    
                    # @var predictions
                    predictions = None


                    # @var model_metadata Dict
                    model_metadata = {
                        'model': best_model,
                        'created_at': time.ctime (os.path.getmtime (model_filename)),
                        'probabilities': predictions
                    }
                    
                    
                    # @var feature_key String
                    feature_key = classifier[0] + '_' + analyzer + '_' + '_'.join (map (str, features__ngram_range))
                    
                    
                    # Store the results
                    if callback:
                        callback (feature_key, y_pred, model_metadata)