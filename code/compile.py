"""
    Compile a dataset

    Following steps:

    (1) To preprocess texts
    ./python -W ignore generate-preprocess-text.py --dataset=...

    (2) For feature NER and POS
    ./python -W ignore generate-ner.py --dataset=...
    
    (3) To generate the features run:
    ./python -W ignore generate-[lf-se-be-bf-...].py --dataset=...
    
    (4) For feature selection
    ./python -W ignore feature-selection.py --dataset=...
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import sys
import config
import bootstrap
import os.path

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


def main ():
    
    # var parser
    parser = DefaultParser (description = 'Compile dataset')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus, args.task, True)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')
    
    
    # @var df Dataframe
    df = dataset.get ()
    

if __name__ == "__main__":
    main ()