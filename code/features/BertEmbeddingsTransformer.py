import sys
import csv
import os.path
import io
import numpy as np
import pandas as pd
import transformers
import torch
from pathlib import Path

from tqdm import tqdm

from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator 


class BertEmbeddingsTransformer (BaseEstimator, TransformerMixin):
    """
    Generate BERT sentence vectors

    @see config.py

    @link https://www.sbert.net/docs/quickstart.html
    """
    
    def __init__ (self, model_path, tokenizer_path = "", cache_file = "", field = "tweet"):
        """
        @param model_path String Path or keyname of the model
        @param tokenizer_path String Path or keyname of the tokenizer
        @param field String
        @param cache_file String
        """
        super ().__init__()
        
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path if tokenizer_path else model_path
        self.field = field
        self.cache_file = cache_file
        self.number_of_features = 768
        self.device = "cuda:0" if torch.cuda.is_available () else "cpu"
    
    
    def mean_pooling (self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        
        @link https://www.sbert.net/examples/applications/computing-embeddings/README.html
        @link https://gist.github.com/haayanau/e7ca837b9503afbf68d1407bed633619
        """
        
        # @var token_embeddings Tensor
        # First element of model_output contains all token embeddings
        # - (batch_size, sequence_length, hidden_size)
        # - torch.Size([35, 38, 768])
        # NOTE: You can return model_output[1] to return the CLS token
        
        token_embeddings = model_output[0]
        
        
        # With Unsqueeze, we change the orientation, and we expand to get the size of the vectors
        # ... [
        #       [1, 1, 1, 0, 0, 0, 0]
        #       [1, 1, 0, 0, 0, 0, 0]
        # ... ]
        # 
        # ... 
        # [
        #   [1 1 1 1 1 ... 1]
        #   [1 1 1 1 1 ... 1]
        #   [1 1 1 1 1 ... 1]
        #   [0 0 0 0 0 ... 0]
        # ], [
        #   [1 1 1 1 1 ... 1]
        #   [1 1 1 1 1 ... 1]
        #   [0 0 0 0 0 ... 0]
        #   [0 0 0 0 0 ... 0]
        #
        #
        input_mask_expanded = attention_mask.unsqueeze (-1).expand (token_embeddings.size ()).float ()
        
        
        # @var sum_embeddings Tensor, Sums the dimension, but with attention mask 1 (sequence_length)
        # torch.Size([35, 768])
        sum_embeddings = torch.sum (token_embeddings * input_mask_expanded, 1)

        
        # @var sum_mask Tensor To bound the values
        # torch.Size([35, 768])
        sum_mask = torch.clamp (input_mask_expanded.sum (1), min = 1e-9)

        
        return sum_embeddings / sum_mask


    def transform (self, X, **transform_params):
    
        # Return vectors from cache
        if self.cache_file and os.path.exists (self.cache_file):
            return pd.read_csv (self.cache_file, header = 0, sep = ",")
        
        
        # @var tokenizer Tokenizer Load the Tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained (self.tokenizer_path)
        
        
        # @var model Model
        model = transformers.AutoModel.from_pretrained (self.model_path)
        model = model.to (self.device)
        
        
        def get_bert_embeddings (df):
            """
            @param df DataFrame
            """

            
            # @var encoded_inputs Get data. Note fillna to avoid None and empty strings
            encoded_inputs = tokenizer (
                df[self.field].fillna ('').tolist (), 
                padding = True, 
                truncation = True, 
                max_length = 256, 
                return_tensors = 'pt'
            ).to (self.device)
            
            
            # Compute token embeddings
            with torch.no_grad ():
                model_outputs = model (**encoded_inputs)
                
                
                sentence_embeddings = self.mean_pooling (model_outputs, encoded_inputs['attention_mask'])
                return pd.DataFrame (sentence_embeddings.cpu ().numpy ())
        
        
        # @var frames List of DataFrames
        frames = []
        
        
        # Iterate on batches
        for chunk in tqdm (np.array_split (X, min ([100, len (X)]))):
            frames.append (get_bert_embeddings (chunk))
        
        
        # @var features DataFrame Concat frames in row axis
        features = pd.concat (frames)
        
        
        # Assign column names
        features.columns = self.get_feature_names ()
        

        # Store
        if self.cache_file:
            features.to_csv (self.cache_file, index = False)
        
        
        # Return vectors
        return features
        
        
    def fit (self, X, y = None, **fit_params):
        return self
        
    def get_feature_names (self):
        return ['be_' + str (x) for x in range (1, self.number_of_features + 1)]
