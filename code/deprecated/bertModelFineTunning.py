"""
    Fine tune a BERT Model for an specific task
    
    @see config.py
    
    @link https://www.sbert.net/docs/quickstart.html
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import csv
import torch
import argparse
import pandas as pd
import numpy as np
import pkbar
import config
import os
import transformers
import sklearn
import utils

import torch.nn.functional as F
import torch.utils.data as data_utils

from datasets import Dataset
from datasetResolver import DatasetResolver
from preprocessText import PreProcessText

from torch.utils.data import DataLoader
from torch import nn



# Parser
parser = argparse.ArgumentParser (description = 'Finetune BERT model')
parser.add_argument ('--dataset', dest = 'dataset', default = next (iter (config.datasets)), help="|".join (config.datasets.keys ()))
parser.add_argument ('--force', dest = 'force', default=False, help="If True, it forces to replace existing files")
parser.add_argument ('--task', dest = 'task', default = '', help = 'Get the task')
parser.add_argument ('--test', dest = 'test', type = lambda x: (str (x).lower () in ['True', 'true','1', 'yes']), default = False, help = 'If true, we evaluate the model with the testing dataset')
parser.add_argument ('--epochs', dest = 'epochs', type = int, default = 4, help = 'Number of epochs for fine-tunning')


# Get args
args = parser.parse_args ()


# Define batch size
batch_size = 64


# Get device
device = torch.device ('cuda') if torch.cuda.is_available () else torch.device ('cpu')


# Defines the transformer tokenizer
def tokenize (batch):
    return tokenizer (batch['tweet'], padding = True, truncation = True)


class BERTModelFineTunning (nn.Module):
    """
    A custom BERT model for fine-tunning an specific task
    """

    def __init__ (self, pretrained_model, num_labels = None):
        super (BERTModelFineTunning, self).__init__()
        self.bert = transformers.BertForSequenceClassification.from_pretrained (pretrained_model, return_dict = True, num_labels = num_labels)
        self.bert.to (device)
        
    def forward (self, input_ids, token_type_ids = None, attention_mask = None, position_ids = None, head_mask = None):
        bert_x = self.bert (input_ids, attention_mask = attention_mask)
        return bert_x.logits


# @var preprocess PreProcessText Used for clean text before tokenization
preprocess = PreProcessText ()


# Iterate over all the tasks
for key, dataset_options in config.datasets[args.dataset].items ():

    # @var dataset_name String Get the dataset name
    dataset_name = args.dataset + '-' + key + '.csv'
    
    
    # @var resolver DatasetResolver Its mission is to determine when to load the generic dataset class or specicialied 
    #                               ones, according to the config
    resolver = DatasetResolver ()
    
    
    # @var dataset Get the most suitable dataset
    dataset = resolver.get (dataset_name, dataset_options, args.force)
    
    
    # @var df DataFrame Get the dataset as a dataframe
    df = dataset.get (args.task)
    df['_label'] = df['label']
    
    
    # @var model_filename String
    model_filename = os.path.join (config.directories['assets'], args.dataset, key, 'bert-finetunning-' + args.task)
    
    
    # @var pretrained_model String Get the most suitable BERT model according to the 
    #                              task type and language
    if args.test:
        pretrained_model = model_filename
    else:
        pretrained_model = 'dccuchile/bert-base-spanish-wwm-uncased' if dataset_options['language'] == 'es' else 'bert-base-uncased'

    
    
    # @var tokenizer_model String
    tokenizer_model = 'dccuchile/bert-base-spanish-wwm-uncased' if dataset_options['language'] == 'es' else 'bert-base-uncased'
    
    
    # @var task_type String Determine the task type
    task_type = dataset.get_task_type (args.task)
    task_type = 'regression'
    
    
    # Preprocess text before tokenize
    if dataset_options['language'] == 'es':
        df['tweet'] = preprocess.expand_acronyms (df['tweet'], preprocess.msg_language)
        df['tweet'] = preprocess.expand_acronyms (df['tweet'], preprocess.acronyms)
    
    for pipe in ['remove_urls', 'remove_mentions', 'remove_digits', 'remove_whitespaces', 'remove_elongations', 'remove_emojis', 'to_lower', 'remove_quotations', 'remove_punctuation', 'remove_whitespaces', 'strip']:
        df['tweet'] = getattr (preprocess, pipe)(df['tweet'])
    
    
    
    # @var num_labels int Get the number of labels to build the BERT classifier
    num_labels = len (df['label'].unique ()) if task_type == 'classification' else 1
    
    
    # @var model BERTModelFineTunning model
    model = BERTModelFineTunning (num_labels = num_labels, pretrained_model = pretrained_model)
    model.to (device)
    
    
    # @var tokenizer Get the pretained tokenizer
    tokenizer = transformers.BertTokenizerFast.from_pretrained (tokenizer_model)
    
    
    # Encode label as numbers instead of user names
    df['label'] = df['label'].astype ('category').cat.codes
    
    # @var TorchDataset Dataset Encode datasets to work with transformers
    TorchDataset = Dataset.from_pandas (df)
    
    
    # Tokenize the dataset
    TorchDataset = TorchDataset.map (tokenize, batched = True, batch_size = len (TorchDataset))
    
    
    # Finally, we "torch" the new columns. We return the rest 
    # of the columns with "output_all_columns"
    TorchDataset.set_format ('torch', columns = ['input_ids', 'attention_mask', 'label'], output_all_columns = True)
    
    
    # Create a dataset with the linguistic features joined, the input id, the attention mask, and the labels
    TorchDataset = data_utils.TensorDataset (
        TorchDataset['input_ids'],
        TorchDataset['attention_mask'],
        TorchDataset['label']
    )
    
    
    # Get datasets splits for training, validation and testing
    train_df = dataset.get_split (df, 0)
    val_df = dataset.get_split (df, 1)
    test_df = dataset.get_split (df, 2)
    
    
    # Generate a sampler from the indexes. 
    # We use this to get the exact samples for training, val, and test 
    # as it is defined for this dataset
    train_sampler = torch.utils.data.SubsetRandomSampler (train_df.index)
    val_sampler = torch.utils.data.SubsetRandomSampler (val_df.index)
    test_sampler = torch.utils.data.SubsetRandomSampler (test_df.index)
    
    
    # Create the loaders
    train_loader = torch.utils.data.DataLoader (TorchDataset, batch_size = batch_size, sampler = train_sampler, shuffle = False)
    val_loader = torch.utils.data.DataLoader (TorchDataset, batch_size = batch_size, sampler = val_sampler, shuffle = False)
    test_loader = torch.utils.data.DataLoader (TorchDataset, batch_size = batch_size, sampler = test_sampler, shuffle = False)
    
    
    # @var optimizer Set the AdamW optimizer
    optimizer = transformers.AdamW (model.parameters (), lr = 2e-5, eps = 1e-8, correct_bias = False)
    
    
    # @var scheduler 
    scheduler = transformers.get_linear_schedule_with_warmup (optimizer, num_warmup_steps = 0, num_training_steps = len (train_loader) * args.epochs)
    
    
    # @var criterion Set the loss Criteria according to the task type
    criterion = torch.nn.CrossEntropyLoss () if task_type == 'classification' else torch.nn.MSELoss ()
    criterion = criterion.to (device)
    
    
    # @var train_per_epoch int
    train_per_epoch = int (len (train_df) / batch_size)
    
    
    # Metrics
    metrics = ['epoch', 'loss', 'acc', 'val_loss', 'val_acc'] if task_type == 'classification' else ['epoch', 'loss', 'val_loss']
    
    
    # @var epochs int If we are on testing mode, we only need one epoch, because we 
    #                 are going only to user the preloaded model to predict
    epochs = 1 if args.test else args.epochs
    
    
    # Iterate by epochs
    for epoch in range (1, epochs + 1):
        
        # Create a progress bar
        # @link https://github.com/yueyericardo/pkbar/blob/master/pkbar/pkbar.py (stateful metrics)
        kbar = pkbar.Kbar (target = train_per_epoch, width = 32, stateful_metrics = metrics)
        
        
        # If we are on fine-tunning mode, we will train our model based on the number of epochs
        if not args.test:
        
            # @var train_loss_set Dict Store our loss for the current epoch
            train_loss_set = []
            
            
            # Store correct predictions for the current epochs
            correct_predictions = 0
            
            
            # Set the model on training mode
            model.train ()
            
            
            # Get all the training batches for this epoch
            for i, (input_ids, attention_mask, labels) in enumerate (train_loader):
            
                # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates 
                # the gradients on subsequent backward passes. This is convenient while training RNNs. 
                # So, the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
                # @link https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                optimizer.zero_grad ()
                
                
                # Move features to device
                input_ids = input_ids.to (device)
                attention_mask = attention_mask.to (device)
                labels = labels.to (device)
                
                
                # @var predictions Forward model
                predictions = model (input_ids, attention_mask = attention_mask)
                
                
                # If the domain is classification, then get the correct predictions
                if task_type == 'classification':
                
                    # Remove dimensions of input of size 1 removed.
                    predictions = torch.squeeze (predictions)
                    
                    
                    # Use max to get the correct class
                    _, preds = torch.max (predictions, dim = 1)
                    
                    
                    # Get the correct predictions
                    correct_predictions += torch.sum (preds == labels)
                    
                    
                    # Get the accuracy
                    acc = correct_predictions.item ()  / (batch_size * (i + 1))
                
                
                    # Get loss
                    loss = criterion (predictions, labels)
                
                # If the problem is regression
                else:
                
                    # Get loss
                    loss = torch.sqrt (criterion (predictions, labels.float ()) + 1e-6)
                
                
                # Store loss
                train_loss_set.append (loss.item ())
                
                
                # Do deep-learning stuff...
                loss.backward ()
                torch.nn.utils.clip_grad_norm_(model.parameters (), 1.0)
                
                
                # Post backward pass network will perform a parameter update optimizer.step function, 
                # based on the current gradient
                optimizer.step ()
                
                
                # Update the learning rate
                scheduler.step ()
                
                
                # Update metrics in each step
                kbar_values = [('epoch', int (epoch)), ("loss", loss.item ())]
                if task_type == 'classification':
                    kbar_values.append (('acc', acc))
                
                kbar.add (1, values = kbar_values)
            
        
        
        # Eval this epoch with the validation test (if we are training) or with the 
        # test set (if we are testing)
        model = model.eval ()
        
        
        # Store correct predictions
        correct_predictions = 0
        
        
        # Store loss
        val_losses = []
        
        
        # Store all predictions in the same order as the loader
        total_predictions = []
        
        
        # Store all labels in the same order as the loader
        total_labels = []
        
        
        # No gradient is needed
        with torch.no_grad ():
            
            # ...
            for i, (input_ids, attention_mask, labels) in enumerate (test_loader if args.test else val_loader):
            
                # Move features to device
                input_ids = input_ids.to (device)
                attention_mask = attention_mask.to (device)
                labels = labels.to (device)
                
                
                # Forward model
                predictions = model (input_ids, attention_mask = attention_mask)
                
                
                # If the domain is classification, then get the correct predictions
                if task_type == 'classification':
                    
                    # Get the predictions
                    predictions = torch.squeeze (predictions)
                    
                    
                    # Use max to get the correct class
                    _, preds = torch.max (predictions, dim = 1)
                    
                    
                    # Get the correct predictions
                    correct_predictions += torch.sum (preds == labels)
                    
                    
                    # Get loss
                    loss = criterion (predictions, labels)
                    
                    
                    # Store for the final report
                    if args.test:
                        total_predictions.extend (preds.detach ().numpy ())
                        total_labels.extend (labels.detach ().numpy ())
                    
                
                # If the problem is regression
                else:
                
                    # Get loss
                    loss = torch.sqrt (criterion (predictions, labels.float ()) + 1e-6)
                    
                    
                    # Store for the final report
                    if args.test:
                        total_predictions.extend (predictions.detach ().numpy ())
                        total_labels.extend (labels.detach ().numpy ())

                
                
                # Get BCE loss
                val_losses.append (loss.item ())

                
            # Update values
            kbar_values = [("val_loss", np.mean (val_losses))]
            if task_type == 'classification':
                kbar_values.append (('val_acc', correct_predictions.item () / (test_df.shape[0] if args.test else val_df.shape[0])))
        
        
            # Update var
            kbar.add (0, values = kbar_values)

    
    # If we were on fine-tuning mode, we need to save the model and the tokenizer
    if not args.test:

        # ...
        os.makedirs (os.path.dirname (model_filename), exist_ok = True)
        

        # Save model
        model.bert.save_pretrained (model_filename)
        tokenizer.save_pretrained (model_filename)
        
    # If we were on testing mode, we need to display the final results
    else:
    
        # Get true labels as string
        true_labels = test_df["_label"].astype ('category')
        true_labels = dict (enumerate (true_labels.cat.categories))
        
        print ("predictions")
        print (total_predictions)
        
        # Transform predictions into labels
        y_predicted_classes = [true_labels[int (prediction)] for prediction in total_predictions]
        y_real_classes =  [true_labels[int (true_label)] for true_label in total_labels]
        
        print ("predicted")
        print (y_predicted_classes)
        
        print ("real")
        print (y_real_classes)
        
        # Classification report
        print (sklearn.metrics.classification_report (y_real_classes, y_predicted_classes))
        
        
        # Confusion matrix
        cm = sklearn.metrics.confusion_matrix (y_real_classes, y_predicted_classes, labels = test_df['_label'].unique (), normalize = 'true')
        
        utils.print_cm (cm, test_df['_label'].unique ())
        
