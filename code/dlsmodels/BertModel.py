"""
    Fine tune a BERT Model for an specific task
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import time
import json
import sys
import csv
import torch
import pandas as pd
import numpy as np
import config
import os
import transformers
import sklearn
import gc
import shutil
import multiprocessing

from pathlib import Path

import ray
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
from ray.tune import Callback

from torch import nn
import torch.nn.functional as F
import torch.utils.data as data_utils

from datasets import Dataset
from torch.utils.data import DataLoader
from torch import nn
from functools import partial

from .BaseModel import BaseModel

import wandb



class BertModel (BaseModel):
    """
    Bert implementation
    """
    
    # @var all_available_features List
    all_available_features = []
    
    
    # @var total_predictions List
    total_predictions = []
    
    
    # @var total_labels List
    total_labels = []
    
    
    # @var total_probabilities List
    total_probabilities = []
    
    
    # @var field String
    field = 'tweet'
    
    
    # @var batch_train_size int Batch size for training
    batch_train_size = 16
    
    
    # @var batch_val_size int Batch size for validating
    batch_val_size = 32
    
    
    # @var tokenizer Get the pretained tokenizer
    tokenizer = None
    
    
    # @var pretrained_model String
    pretrained_model = ''
    
    
    @classmethod
    def adapt_argument_parser (cls, parser):
        parser.add_argument ('--pretrained-model', dest = 'pretrained_model', default = '', help = 'Select the pretrained model to train (HuggingFace)')


    def set_arguments_parser (self, args):
        """
        @param args
        """
        self.pretrained_model = args.pretrained_model    

    
    
    def get_folder (self):
        """
        @inherit
        """
        return self.folder or os.path.join ('transformers-' + self.get_pretrained_model ().replace ('/', '-'))
    
    
    def get_model_filename (self):
        """
        @return String
        """
        return self.dataset.get_working_dir (self.dataset.task, 'models', self.get_folder ())
    
    
    def set_pretrained_model (self, pretrained_model):
        self.pretrained_model = pretrained_model
        
        
    def get_pretrained_model_per_language (self, lang):
        if self.dataset.get_dataset_language () == 'es':
            return [
                'PlanTL-GOB-ES/roberta-base-bne', 
                'PlanTL-GOB-ES/roberta-large-bne',
                'dccuchile/bert-base-spanish-wwm-cased',
                'dccuchile/bert-base-spanish-wwm-uncased',
                'bert-base-multilingual-uncased',
                'bert-base-multilingual-cased',
                'bertin-project/bertin-roberta-base-spanish'
            ]
    
        if self.dataset.get_dataset_language () == 'en':
            return [
                'bert-base-cased', 
                'bert-base-uncased', 
                'roberta-base',
                'bert-base-multilingual-uncased',
                'bert-base-multilingual-cased',
                'bertin-project/bertin-roberta-base-spanish'
            ]
    
    def get_pretrained_model (self):
        """
        Get the most suitable BERT model according to the task type and language
        
        @return String
        """
        
        if not self.pretrained_model:
            self.pretrained_model = self.get_suggested_pretrained_model ()
        
        return self.pretrained_model
    
    
    def get_suggested_pretrained_model (self):
        """
        @return String
        """
        
        if self.dataset.get_dataset_language () == 'es':
            return 'PlanTL-GOB-ES/roberta-base-bne'
            
        elif self.dataset.get_dataset_language () == 'en': 
            return 'bert-base-uncased'

        elif self.dataset.get_dataset_language () == 'ar': 
            return 'asafaya/bert-base-arabic'
            
        else:
            return 'bert-base-multilingual-cased'
            
    
    def get_tokenizer_filename (self):
        """
        @return String
        """
        return self.get_pretrained_model ()

    
    def tokenize (self, batch):
        """
        @param batch
        """
        return self.tokenizer (batch[self.field], padding = True, truncation = True)


    def createDatasetFromPandas (self, df, task_type = 'classification'):
        """
        Encode datasets to work with transformers from a DataFrame and 
        "torch" the new columns. 
        
        @param df DataFrame
        @param task_type String
        
        @return Dataset
        """
        
        # @var label_field String
        label_field = 'label'
        
        
        if 'classification' == task_type:
            df['_temp'] = df[label_field].astype (str).astype ('category').cat.codes

        elif 'regression' == task_type:
            df['_temp'] = df[label_field]

        elif 'multi_label' == task_type:

            # Create a binarizer
            lb = sklearn.preprocessing.MultiLabelBinarizer ()
            
            
            # Adjust label
            df[label_field] = df[label_field].astype ('category')
            df[label_field] = df[label_field].cat.add_categories ('none')
            df[label_field].fillna ('none', inplace = True)
            
            
            # Fit the multi-label binarizer
            lb.fit ([self.dataset.get_available_labels () + ["none"]])
            
            
            # Encode the labels and merge to the dataframe
            # Note we use "; " to automatically trim the texts
            temp_values = pd.DataFrame (lb.transform (df[label_field].str.split ('; ')), index = df.index, columns = lb.classes_).astype (np.float64).values.tolist ()
            df['_temp'] = temp_values
            
            
            # Note the change of the name
            label_field = 'labels'
        
        # Get the only labels we care (labels -if any- and the text)
        df = df[[self.field, '_temp']].rename (columns = {'_temp': label_field})
        
        
        # @var dataset Dataset
        dataset = Dataset.from_pandas (df)
        
        
        # Get the input_ids and the attention mask
        dataset = dataset.map (partial (self.tokenize), batched = True, batch_size = len (dataset))
        
        
        # Torch (transform into tensors) the input ids, the attention mask, and the label
        dataset.set_format ('torch', columns = ['input_ids', 'attention_mask', label_field], output_all_columns = False)
        
        
        # Return the result
        return dataset
        
        
    def compute_metrics (self, pred):
        """
        
        This function allows to calculate accuracy, precision, recall, and f1
        for the Trainer huggingface component
        
        @todo Adapt to another classifying problems such as regression or 
              binary
        
        return Dict
        """
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
        
        # @var labels
        labels = pred.label_ids
        
        
        # @var preds
        preds = pred.predictions.argmax (-1) if 'classification' == task_type else pred.predictions
        
        
        # Store metrics in the class
        self.total_predictions = preds
        self.total_labels = labels
        self.total_probabilities = pred.predictions


        # Round the predictions
        if 'multi_label' == task_type:
            preds = np.rint (preds)
        
        
        # Classification metrics
        if task_type in ['classification', 'multi_label']:
        
            # @var precision Float @var recall Float @var f1 Float @var _support array
            precision, recall, f1, _support = sklearn.metrics.precision_recall_fscore_support (labels, preds, average = 'macro')
            
            
            return {
                'f1': f1,
            }

        
        # Regression
        else:
        
            return {}
    
    def train (self, using_official_test = True, force = False):
        """
        @inherit
        
        @param using_official_test Boolean
        @param force Boolean
        """
    
        # @var df DataFrame
        df = self.dataset.get ()
        
        
        # @var val_split String 
        # Determine which split do we get based on if we 
        # want to validate the val dataset of the train dataset
        val_split = 'val' if not using_official_test else 'test'
        
        
        # @var model_folder String
        model_folder = self.get_folder ()
        
        
        # @var model_filename String
        model_filename = self.get_model_filename ()
        
        
        # @var hyperparameters_file_path String
        hyperparameters_file_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'hyperparameters.csv')
        
        
        # @var epochs_to_evaluate List
        epochs_to_evaluate = [1, 2, 3, 4, 5]
        
        
        # @var n_trials int
        n_trials = 10
        
        
        # @var default_num_train_epochs int
        default_num_train_epochs = 3
        
        
        # @var default_warmup_steps int
        default_warmup_steps = 500
        
        
        # @var default_weight_decay float
        default_weight_decay = 0.01
        
        
        # @var tokenizer_filename String
        tokenizer_filename = self.get_tokenizer_filename ()
        
        
        # @var training_args TrainingArguments
        training_args = transformers.TrainingArguments (
            report_to = "wandb",
            run_name = model_folder,
            save_strategy = "epoch",
            save_total_limit = 1,
            evaluation_strategy = 'epoch',
            eval_steps = 500,
            disable_tqdm = True,
            load_best_model_at_end = True,
            output_dir = './results',
            logging_dir = './logs'
        )
        
        
        # @var task_type String
        task_type = self.dataset.get_task_type ()
        
        
        # @var resume_file_path String
        resume_file_path = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'training_resume.json')


        # @var resources_per_trial Dict
        if torch.cuda.is_available ():
            resources_per_trial = {
                "gpu": 1
            }
        else:
            resources_per_trial = {
                "cpu": min (8, multiprocessing.cpu_count ())
            }
        
        
        # @var train_resume 
        train_resume = {
            'model': 'transformers',
            'iterations': n_trials,
            'folder': model_folder,
            'dataset': self.dataset.dataset,
            'corpus': self.dataset.corpus,
            'task': self.dataset.task,
            'task_type': task_type,
            'labels': self.dataset.get_num_labels (),
            'pretrained_model': self.get_pretrained_model (),
            'warmup_steps': default_warmup_steps,
            'weight_decay': default_weight_decay,
            'batch_train_size': self.batch_train_size,
            'batch_val_size': self.batch_val_size,
            'tokenizer_field': self.field,
            'tokenizer_model': tokenizer_filename,
            'epochs': epochs_to_evaluate,
            'resources_per_trial': resources_per_trial
        }
        
        
        # Store
        with open (resume_file_path, 'w') as resume_file:
            json.dump (train_resume, resume_file, indent = 4, sort_keys = True)
        
    
        # Load the tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained (tokenizer_filename)
        
        
        # @var train_df DataFrame Get training split
        train_df = self.dataset.get_split (df, 'train')
        
        
        # @var val_df DataFrame Get validation split
        val_df = self.dataset.get_split (df, val_split)

        
        # @var train_dataset Dataset Encode datasets to work with transformers
        train_dataset = self.createDatasetFromPandas (train_df, task_type)
        
        
        # @var val_dataset Dataset Encode datasets to work with transformers
        val_dataset = self.createDatasetFromPandas (val_df, task_type)
        
        
        # @var model BertForSequenceClassification Loads your fine-tunned model
        # ...
        def model_init ():
            # @var task_type String
            task_type = self.dataset.get_task_type ()
            
            
            # @var problem_type String
            problem_type =  "multi_label_classification" if task_type == 'multi_label' else None
            
            
            # @var num_labels int
            num_labels = self.dataset.get_num_labels ()
            if task_type == 'multi_label':
                num_labels = num_labels + 1
            
            
            return transformers.AutoModelForSequenceClassification.from_pretrained (
                self.get_pretrained_model (), 
                problem_type = problem_type,
                num_labels = num_labels
            )
        
        
        # @var trainer
        trainer = transformers.Trainer (
            args = training_args, 
            train_dataset = train_dataset, 
            eval_dataset = val_dataset,
            compute_metrics = self.compute_metrics,
            model_init = model_init
        )
        
        
        # @var hp_space Dict
        hp_space = {
            'weight_decay': tune.uniform (0.0, 0.3) if n_trials > 1 else default_weight_decay,
            'per_device_train_batch_size': tune.choice ([8, 16])  if n_trials > 1 else self.batch_train_size,
            'warmup_steps': tune.choice ([0, 250, 500, 1000]) if n_trials > 1 else default_warmup_steps,
            'num_train_epochs': tune.choice (epochs_to_evaluate) if n_trials > 1 else default_num_train_epochs,
        }

        if n_trials > 1:
            hp_space['learning_rate'] = tune.uniform (1e-5, 5e-5)
        
        
        @ray.remote
        class HyperparametersDataFrame:
            
            # @var hyperparameter_df None
            hyperparameter_df = None
            
            def add (self, row):
                
                # @var temp_df DataFrame
                temp_df = pd.DataFrame.from_records ([row], index = 'trial_id')

                # Attach the row
                if self.hyperparameter_df is None:
                    self.hyperparameter_df = temp_df
                
                else:
                    self.hyperparameter_df = pd.concat ([self.hyperparameter_df, temp_df])

            def update_metric (self, trial_id, metric, objective):
                self.hyperparameter_df.loc[trial_id, metric] = objective

            def get (self):
                return self.hyperparameter_df


        # @var hyperparameter_df_handler 
        hyperparameter_df_handler = HyperparametersDataFrame.remote ()
        
        
        class GetBestTrialCallback (Callback):
            """
            GetBestTrialCallback
            
            Determines which was the best run. This is needed to 
            delete the rest of the models after training
            """

            # @var best_id 
            best_id = None
            
            
            # @var best_result int The best metric found 
            best_result = 0
            
            
            def __init__ (self, hyperparameter_df_handler):
                self.hyperparameter_df_handler = hyperparameter_df_handler
                
            def on_trial_start (self, iteration, trials, trial, **info):

                # @var trial_parameters Dict
                trial_parameters = {}
                
                
                # Set parameters
                trial_parameters['trial_id'] = trial.trial_id
                trial_parameters['objective'] = 0
                trial_parameters['best'] = False
                trial_parameters = {**trial_parameters, **trial.evaluated_params}

                
                # Attach
                self.hyperparameter_df_handler.add.remote (trial_parameters)
                
            def on_trial_result (self, iteration, trials, trial, result, **info):
            
                if 'done' in result:
                
                    # Update the result in the global dataframe
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'objective', result['objective'])
                    self.hyperparameter_df_handler.update_metric.remote (trial.trial_id, 'time_this_iter_s', result['time_this_iter_s'])
                    
                
                    # Report result
                    print ("Iteration {iteration} finished. Result: {objective}".format (
                        iteration = iteration, 
                        objective = result['objective']
                    ))

        
        
        # @var callback GetBestTrialCallback
        callback = GetBestTrialCallback (hyperparameter_df_handler)


        # Init WANDB
        wandb.init (project = model_folder, entity = "joseagd")
        
        
        # @var analysis
        analysis = trainer.hyperparameter_search (
            local_dir = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder),
            name = 'hyperpameters',
            direction = "maximize", 
            backend = "ray", 
            search_alg = HyperOptSearch (metric = "objective", mode = "max"),
            scheduler = ASHAScheduler (metric = "objective", mode = "max"),
            n_trials = n_trials,
            keep_checkpoints_num = 1,
            checkpoint_score_attr = "training_iteration",
            resources_per_trial = resources_per_trial,
            hp_space = lambda _: hp_space,
            checkpoint_at_end = False,
            verbose = 0,
            fail_fast = False,
            callbacks = [callback]
        )
        
        
        # @var hyperparameter_df DataFrame
        hyperparameter_df = ray.get (hyperparameter_df_handler.get.remote ()).copy (deep = True)
        
        
        # @var best_run_id String
        best_run_id = hyperparameter_df['objective'].idxmax ()
        
        
        # Store the maximum value
        hyperparameter_df.loc[best_run_id, 'best'] = True
        
        
        # Show hyperparameters
        print (hyperparameter_df)
        hyperparameter_df.to_csv (hyperparameters_file_path, index = False)
        
        
        # Remove unused folders
        # Keep the best model parameters
        p = Path (self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'hyperpameters'))
        for directory in [x for x in p.iterdir () if x.is_dir ()]:
        
            # Best run
            if best_run_id in str (directory):
                
                print ("best_run")
                
                # For all directories
                for checkpoint_level_1_dir in os.listdir (str (directory)):
                
                    # Check if is a checkpoint
                    if (str (checkpoint_level_1_dir).startswith ('checkpoint_')):
                    
                        # @var checkpoint_level_1_dir_path Update
                        checkpoint_level_1_dir_path = os.path.join (directory, str (checkpoint_level_1_dir))
                    
                        print ("-----------------")
                        print (checkpoint_level_1_dir_path)
                        print ("-----------------")
                        
                    
                        # Level 2
                        for checkpoint_level_2_dir in os.listdir (checkpoint_level_1_dir_path):
                        
                            # @var checkpoint_level_2_dir_path Update
                            checkpoint_level_2_dir_path = os.path.join (str (directory), str (checkpoint_level_1_dir), str (checkpoint_level_2_dir))
                            
                            
                            # ...
                            if (str (checkpoint_level_2_dir).startswith ('checkpoint-')):
                                for file in os.listdir (checkpoint_level_2_dir_path):
                                    file_path = os.path.join (str (directory), str (checkpoint_level_1_dir), str (checkpoint_level_2_dir), str (file))
                                    shutil.copyfile (file_path, self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, os.path.basename (file)))
            
        
            shutil.rmtree (directory, ignore_errors = True)


        # Update
        train_resume['run_id'] = str (analysis.run_id)
        train_resume['original_pretrained_model'] = train_resume['pretrained_model']
        train_resume['pretrained_model'] = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder)

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
        
        
        # @var _model_file String
        resume_file = self.dataset.get_working_dir (self.dataset.task, 'models', model_folder, 'training_resume.json')
        
        
        # @var training_info Dict
        training_info = self.retrieve_training_info (resume_file)
        
        
        # @var model_filename String Give priority to the one used during training
        model_filename = training_info['pretrained_model'] if 'pretrained_model' in training_info else self.get_model_filename ()
        
        
        # @var tokenizer_filename String Give priority to the one used during training
        tokenizer_filename = training_info['tokenizer_model'] if 'tokenizer_model' in training_info else self.get_tokenizer_filename ()
        
        
        # @var num_labels int 
        num_labels = training_info['labels'] if 'labels' in training_info else self.dataset.get_num_labels ()
        
        
        # @var model BertForSequenceClassification
        model = transformers.AutoModelForSequenceClassification.from_pretrained (model_filename, num_labels = num_labels)


        # @var task_type string
        task_type = training_info['task_type'] if 'task_type' in training_info else self.dataset.get_task_type ()
        
        
        # @var tokenizer Get the pretained tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained (tokenizer_filename)
        
        
        # Set the model in evaluation form in order to get reproductible results
        model.eval ()
        
        
        # @var hugging_face_dataset Dataset Encode datasets to work with transformers
        hugging_face_dataset = self.createDatasetFromPandas (df)
        
        
        # @var batch_val_size int
        batch_val_size = training_info['batch_val_size'] if 'batch_val_size' in training_info else self.batch_val_size
        
        
        # @var training_args TrainingArguments
        training_args = transformers.TrainingArguments (
            output_dir = './results',
            num_train_epochs = 1,
            per_device_train_batch_size = 0,
            per_device_eval_batch_size = batch_val_size,
            warmup_steps = 500,
            weight_decay = 0.01,
            logging_dir = './logs',
        )
        
        
        # @var trainer Trainer Get the trainer, but only for evaluating purposes
        trainer = transformers.Trainer (
            model = model, 
            args = training_args, 
            eval_dataset = hugging_face_dataset, 
            compute_metrics = self.compute_metrics
        )
        
        
        # @var predictions @todo I think I should return this
        predictions = trainer.predict (hugging_face_dataset)
        
        
        # Transform predictions into labels
        if 'classification' == task_type:

            # @var true_labels Get true labels as string
            true_labels = self.dataset.get_true_labels ()
            
            
            # @var y_predicted_classes
            y_predicted_classes = [true_labels[int (prediction)] for prediction in self.total_predictions]
            
        
        
        # @var model_metadata Dict
        model_metadata = {
            'model': model,
            'created_at': time.ctime (os.path.getmtime (model_filename)) if os.path.isfile (model_filename) else None,
            'probabilities': self.total_probabilities
        }
        
        
        # run callback
        if callback:
            callback ('bert', y_predicted_classes, model_metadata)
        
        
