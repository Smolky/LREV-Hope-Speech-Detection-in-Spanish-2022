"""
    To do random stuff
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import config
import argparse
import pandas as pd
import numpy as np
import pickle

from tqdm import tqdm
from dlsmodels.ModelResolver import ModelResolver
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver

from features.LinguisticFeaturesTransformer import LinguisticFeaturesTransformer
from features.SentenceEmbeddingsTransformer import SentenceEmbeddingsTransformer
from features.BertEmbeddingsTransformer import BertEmbeddingsTransformer
from features.TokenizerTransformer import TokenizerTransformer



def main ():

    # var parser
    parser = DefaultParser (description = 'To do random stuff')
    
    
    # @var model_resolver ModelResolver
    model_resolver = ModelResolver ()

    
    # var parser
    parser.add_argument ('--model', 
        dest = 'model', 
        default = model_resolver.get_default_choice (), 
        help = 'Select the family or algorithms to evaluate', 
        choices = model_resolver.get_choices ()
    )
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df El dataframe original (en mi caso es el dataset.csv)
    df = dataset.get ()


    # @var language String
    language = dataset.get_dataset_language ()


    """
    # @var lf_transformers LF
    lf_transformers = LinguisticFeaturesTransformer (cache_file = dataset.get_working_dir ('lf-test.csv'))
    print (lf_transformers.transform (dataset.get_split (df, 'test')))
    """

    """
    # @var fasttext_model SE
    fasttext_model = config.pretrained_models[language]['fasttext']['binary']
    se_transformers = SentenceEmbeddingsTransformer (fasttext_model, cache_file = dataset.get_working_dir ('se-test.csv'), field = 'tweet_clean_lowercase')
    print (se_transformers.transform (dataset.get_split (df, 'test')))
    """

    
    # @var huggingface_model String
    # @todo Redone this after training a finetuned BERT model
    """
    if language == 'es':
        huggingface_model = 'dccuchile/bert-base-spanish-wwm-uncased'
    else:
        huggingface_model = 'bert-base-uncased'
    
        
    be_transformers = BertEmbeddingsTransformer (huggingface_model, cache_file = dataset.get_working_dir ('be-test.csv'), field = 'tweet_clean_lowercase')
    print (be_transformers.transform (dataset.get_split (df, 'test')))
    """
    
    """
    # @var we_transformers WE
    transformer = TokenizerTransformer (cache_file = dataset.get_working_dir ('we-test.csv'), field = 'tweet_clean_lowercase')
    transformer.load_tokenizer_from_disk (dataset.get_working_dir ('we_tokenizer.pickle')) 
    print (transformer.transform (dataset.get_split (df, 'test')))
    """

    """
    # @var cache_file String
    cache_file = dataset.get_working_dir ('bf-test.csv')
    

    # @var huggingface_model String
    huggingface_model = dataset.get_working_dir (dataset.task, 'models', 'bert', 'bert-finetunning')
    

    # Create 
    be_transformers = BertEmbeddingsTransformer (
        huggingface_model, 
        cache_file = cache_file, 
        field = 'tweet_clean_lowercase'
    )

    print (be_transformers.transform (dataset.get_split (df, 'test')))
    """
    
    for feature in tqdm (['lf', 'se', 'bf', 'we', 'be']):
        train_val_df = pd.read_csv (dataset.get_working_dir (feature + '.csv'), header = 0, sep = ",")
        test_df = pd.read_csv (dataset.get_working_dir (feature + '-test.csv'), header = 0, sep = ",")
        
        features_df = pd.concat ([train_val_df, test_df], axis = 0).reset_index (drop = True)
        features_df.to_csv (dataset.get_working_dir (feature + '.csv'), index = False)
        
    
    
if __name__ == "__main__":
    main ()
