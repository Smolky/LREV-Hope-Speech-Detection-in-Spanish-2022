import os
import pandas as pd
import sys
import json
import numpy as np

from sklearn.metrics import f1_score
from sklearn.utils.extmath import weighted_mode

from pathlib import Path
from features.FeatureResolver import FeatureResolver
from .ModelResolver import ModelResolver
from .BaseModel import BaseModel

class EnsembleModel (BaseModel):

    # @var models List
    models = []


    # @var strategy String
    strategy = 'mean'


    """
    Ensemble Model
    
    This model uses the predictions of the other models
    
    """
    
    def get_folder (self):
        """
        @inherit
        """
        return self.folder or 'ensemble' + '-' + self.strategy + '-' + ('-'.join (self.models))
        
        
    def clear_session (self):
        """
        clear_session
        """
        super ()
        self.models = [];
        
    def add_model (self, model):
        """
        @inherit
        """
        return self.models.append (model)
        
        
    def set_ensemble_strategy (self, strategy):
        self.strategy = strategy
        
    
    def train (self, force = False, using_official_test = True):
        """
        @inherit
        """
    
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var model_folder String
        model_folder = self.get_folder ()
        
        
        # @var resume_file_path String
        resume_file_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'training_resume.json')
        
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
        
        # @var train_resume 
        train_resume = {
            'model': 'ensemble',
            'folder': model_folder,
            'dataset': self.dataset.dataset,
            'corpus': self.dataset.corpus,
            'task': self.dataset.task,
            'task_type': task_type,
            'labels': self.dataset.get_num_labels (),
            'models': self.models,
            'strategy': self.strategy
        }
        
        
        # Store
        with open (resume_file_path, 'w') as resume_file:
            json.dump (train_resume, resume_file, indent = 4, sort_keys = True)
        
        
    def predict (self, using_official_test = False, callback = None):
        """
        @inherit
        """
        
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var model_folder String
        model_folder = self.get_folder ()
        
        
        # @var model_resolver ModelResolver
        model_resolver = ModelResolver ()
        
        
        # @var _model_file String
        resume_file = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'training_resume.json')
        
        
        # @var training_info Dict
        training_info = self.retrieve_training_info (resume_file)
        
        
        # @var result_predictions Dict To store predictions of each model / feature set
        result_predictions = {}
        
        
        # @var result_probabilities Dict To store probabilities of each model / feature set
        result_probabilities = {}
        
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
            
        # @var number_of_classes int
        number_of_classes = self.dataset.get_num_labels ()


        # @var labels List
        labels = self.dataset.get_available_labels ();

    
        # Determine if the task is binary or multi-class
        is_binary = number_of_classes <= 2
        

        
        def callback_ensemble (feature_key, y_pred, model_metadata):
        
            """
            This callback is used to keep track of the result of each model
            separately
            
            @param feature_key String
            @param y_pred List
            @param model_metadata Dict
            """
            
            # Store predictions
            result_predictions[feature_key] = pd.DataFrame (
                {feature_key: y_pred} if task_type == 'classification' else y_pred,
                index = df.index
            )

            # @var columns List
            columns = [feature_key + "_" + _label for _label in self.dataset.get_available_labels ()] \
                if task_type == 'classification' else feature_key
            
            
            # For binary classification problems
            # We are assuming that the positive class is the first, but this is not always 
            # true
            if self.dataset.get_num_labels () <= 2:
            
                if 'non-' in columns[0] :
                    columns = [columns[1], columns[0]]
                else:
                    columns = [columns[0], columns[1]]

            
            # Store probabilities
            result_probabilities[feature_key] = pd.DataFrame (
                model_metadata['probabilities'], 
                columns = columns, 
                index = df.index
            )

        
        
        # Iterate over ensemble models
        for submodel_folder in training_info['models']:
        
            # @var submodel_training_resume_file String
            submodel_training_resume_file = self.dataset.get_working_dir (self.dataset.task, 'models', submodel_folder, 'training_resume.json')
            
            
            print (submodel_folder)
            print ("-------------")
            
            
            # Retrieve the submodel information
            if os.path.isfile (submodel_training_resume_file):
                with open (submodel_training_resume_file) as json_file:
                    submodel_training_resume = json.load (json_file)
            else:
                print ("model not found: {path}".format (path = submodel_training_resume_file))
                sys.exit ()
            
            
            
            # @var ensemble_model Model Configure each model of the ensemble
            ensemble_model = model_resolver.get (submodel_training_resume['model'])
            ensemble_model.set_dataset (self.dataset)
            ensemble_model.is_merged (self.dataset.is_merged)
            ensemble_model.set_folder (submodel_folder)
            ensemble_model.clear_session ()
            
            
            # Evaluate models with external features, that is, features from transformers
            if ensemble_model.has_external_features ():
            
                # @var feature_resolver FeatureResolver
                feature_resolver = FeatureResolver (self.dataset)
                
                
                # @var feature_combinations List
                feature_combinations = submodel_training_resume['features'] if 'features' in submodel_training_resume else {}
                
                
                # Iterate over all available features
                for feature_set, features_cache in feature_combinations.items ():
                
                    # Indicate what features are loaded
                    print ("\t" + features_cache)
                    if not Path (features_cache).is_file ():
                        print ("features not found: {features}".format (features = features_cache))
                        sys.exit ()
                        
                    
                    # Set features
                    ensemble_model.set_features (feature_set, feature_resolver.get (feature_set, cache_file = features_cache))


                # Perform the prediction
                ensemble_model.predict (using_official_test = using_official_test, callback = callback_ensemble)
            
            # Models with no external features
            else:
            
                # Perform the prediction
                ensemble_model.predict (using_official_test = using_official_test, callback = callback_ensemble)
            
                
        # @var concat_df Dataframe The ensemble composed by 
        ensemble_df = pd.concat (result_predictions.values (), axis = 'columns')
        
        
        # @var y_pred List
        y_pred = None
        
        
        # @var weights 
        weights = None
        
        
        # Classification ensembles
        if task_type == 'classification':
        
            # Weighted strategy. Soft voting
            if 'weighted' == self.strategy:
        
                # @var weights List|None
                weights = None

        
                if self.dataset.default_split == 'test':
                    weights = pd.read_csv (self.dataset.get_working_dir (self.dataset.task, 'results', 'val', 'ensemble', 'ensemble-weighted-' + '-'.join (self.features_in_the_ensemble), 'weights.csv')).to_dict (orient='list')
                    weights = {key: weight[0] for key, weight in weights.items ()}
                
                else:
                    weights = {
                        feature: f1_score (
                            y_true = self.dataset.get ()['label'], 
                            y_pred = ensemble_df[feature], 
                            average = 'weighted'
                        ) for feature in ensemble_df.columns
                    }
                    
                    
                    # Normalize to 0 ... 1 scale
                    weights = {key: (weight / sum (weights.values ())) for key, weight in weights.items ()}
                    
                    
                # @var weights Dict Filter only the weights of the features we are interested in
                weights = {key: weight for key, weight in weights.items () if key in self.features_in_the_ensemble}
            
            
                # @var y_pred Soft voting ensemble
                y_pred = ensemble_df[self.models].apply (lambda row: weighted_mode (row, list (weights.values ()))[0][0], axis=1).to_list ()
            
            
            # Mean strategy
            if 'mean' == self.strategy:
                
                # @var y_pred Mean probabilities
                y_pred = pd.concat (result_probabilities, axis = 1)
                
                
                # Average probabilities by label
                y_pred = pd.concat (
                    [y_pred.iloc[:, t::len (labels)].mean (axis = 1) for t in range (len (labels))], 
                    axis = 1,
                )
                y_pred.columns = labels
                print (y_pred)


                
                y_pred = np.argmax (y_pred.values, axis = 1)
                y_pred = [labels[int (item)] for item in y_pred]
            

            # Mode strategy Ensemble based on the mode (hard voting) 
            if 'mode' == self.strategy:
                y_pred = ensemble_df[self.models].mode (axis = 'columns')[0]


            # Highest strategy.
            if 'highest' == self.strategy:
                y_pred = pd.concat (result_probabilities, axis = 1)

                # Binary
                if is_binary:
                    y_pred = y_pred.max (axis = 1)
                    y_pred = y_pred > .5
                    y_pred = [labels[int (item)] for item in y_pred]
                
                # Multiclass
                else:
                    y_pred = y_pred.idxmax (axis = 1)
                    y_pred = [item[1].split ('_')[1] for item in y_pred]


            # @var probabilities List
            probabilities = []
            
            
            # @var merged_probabilities DataFrame
            merged_probabilities = pd.concat (result_probabilities.values (), axis = 1)
            
            
            # Iterate...
            for idx, y_pred_item in enumerate (y_pred):
            
                # @var labels Series
                labels = ensemble_df[self.models].iloc[idx]
            
                
                # @var feature_sets List
                feature_sets = [feature_set for feature_set, label in labels.iteritems () if label == y_pred_item]
                
                
                # @var temp Dict
                temp = {}
            
            
                #Iterate over each label
                for label in self.dataset.get_available_labels ():
                    
                    # @var cols List Retrieve the labels that match the matching class
                    cols = [col for col in merged_probabilities \
                        if col.startswith (tuple (feature_sets)) and col.endswith ('_' + label)]
                    
                    
                    # Calculate values
                    temp[label] = merged_probabilities.iloc[idx][cols].mean ()


                # Attach probabilities for each label
                probabilities.append (list (temp.values ()))
        
        
            # @var model_metadata Dict We get information of the model that 
            # we can use in the main callback
            model_metadata = {
                'model': None,
                'created_at': '',
                'probabilities': probabilities,
                'weights': weights
            }
            
            
            # Run the main callback
            callback (
                feature_key = self.get_folder (), 
                y_pred = y_pred, 
                model_metadata = model_metadata
            )
            
        

        # Regression tasks
        if task_type == 'regression':
            
            # @todo
            print ("@todo")
            sys.exit ()
            
            # print (ensemble_df)
            # @var y_pred_mode is the mode
            # y_pred_mode = ensemble_df[self.models].mean (axis = 'columns')[0]        
        
            print (y_pred_mode)
            print ("@todo")
            sys.exit ()