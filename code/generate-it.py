"""
    Generate Image Tags
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser

from features.ImageTagsTransformer import ImageTagsTransformer


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Generate Image Tags')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, args.force)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    
    # @var df Dataframe
    df = dataset.get ()
    
    
    # @var language String
    language = dataset.get_dataset_language ()

    
    # @var vision_model String
    vision_model = config.computer_vision['yolo']
    
    
    # @var it_transformers ImageTagsTransformer
    it_transformers = ImageTagsTransformer (vision_model, 
        cache_file = dataset.get_working_dir (args.task, 'it.csv'),
        image_path = dataset.get_working_dir ('images'),
    )
    
    print (it_transformers.transform (df))
    

if __name__ == "__main__":
    main ()