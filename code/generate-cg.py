"""
    Generate character n-grams features
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from features.NGramsTransformer import NGramsTransformer


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate character n-grams (CG) features')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var train_df DataFrame Get training split
    train_df = dataset.get_split (df, 'train')
    
    
    # @var cache_file String
    cache_file = dataset.get_working_dir (args.task, 'cg.csv')
    
    
    # @var cg_transformers NG
    cg_transformers = NGramsTransformer (
        analyzer = 'char',
        cache_file = cache_file
    )
    cg_transformers.fit (train_df)
    cg_transformers.save_vectorizer_on_disk (dataset.get_working_dir ('cg_vectorizer.pickle'))


    # @var vectorizer VectorizerTransformer
    vectorizer = cg_transformers.get_vectorizer ()
    
    
    # Print the embeddings generated
    print (cg_transformers.transform (df))    
    

    

if __name__ == "__main__":
    main ()