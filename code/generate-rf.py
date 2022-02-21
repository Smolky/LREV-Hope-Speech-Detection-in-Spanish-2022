"""
    Generate RoBERTa embeddings from the fine-tuned model
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import json

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.RobertaEmbeddingsTransformer import RobertaEmbeddingsTransformer


def main ():

    # var parser
    parser = DefaultParser (description = 'Generate RoBERTa embeddings from the finetuned model')
    
    
    # Add parser
    parser.add_argument ('--folder', dest = 'folder', default = '', help = 'Select a name of the model')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()
    
    
    # @var cache_file String
    cache_file = dataset.get_working_dir (dataset.task, 'rf.csv')


    # @var training_resume_file String
    training_resume_file = dataset.get_working_dir (dataset.task, 'models', args.folder, 'training_resume.json')
    
    
    # Normal behaviour. The resume file exists
    if os.path.isfile (training_resume_file):
        
        with open (training_resume_file) as json_file:
            training_resume = json.load (json_file)

    # For keeping compatibility
    else:
        print ("Can't find the training resume file")
        sys.exit ()


    # @var model_type String
    model_type = training_resume['model'] if 'model' in training_resume else 'deep-learning'
    
    
    # Check
    if model_type != 'transformers':
        print ("A transformer's based model is required to build the RoBERTa fine-tunned features")
        sys.exit ()
    
    
    # @var pretrained_model String
    pretrained_model = training_resume['pretrained_model']


    # @var pretrained_model String
    tokenizer_path = training_resume['tokenizer_model']
    

    # @var be_transformers BertEmbeddingsTransformer
    rf_transformers = RobertaEmbeddingsTransformer (
        pretrained_model, 
        tokenizer_path = tokenizer_path,
        cache_file = cache_file, 
        field = 'tweet'
    )

    print (rf_transformers.transform (df))

    
if __name__ == "__main__":
    main ()
