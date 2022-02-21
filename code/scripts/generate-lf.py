"""

"""

import bootstrap
import sys
import config

from datasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


def main ():
    
    # var parser
    parser = DefaultParser (description = 'For testing')
    

    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var resolver Resolver
    resolver = DatasetResolver ()
    
    
    # @var dataset Dataset
    dataset = resolver.get (args.dataset, args.corpus)


    # @var df Dataframe
    df = dataset.get ()
            
            
        # Preprorcess the data
        df = dataset.preprocess (df, ['remove_whitespaces'])



        # Get precached pos tagger
        postagger_filename = os.path.join (config.directories['assets'], args.dataset, key, args.dataset + "-" + key + "-tagger.csv")
        
        
        # Load POS tagger
        if os.path.exists (postagger_filename):
            
            # Get dataframe
            pos_tagger_df = pd.read_csv (postagger_filename, header = 0, sep = ",")
            
            
            # Combine with dataframe
            df = df.assign (tagged_pos = pos_tagger_df['tagged_pos'])
            df = df.assign (tagged_ner = pos_tagger_df['tagged_ner'])
            
            
        # Iterate in batches
        for idx, df_batch in enumerate (batch (df, batch_number = batch_size)):
        
            if idx >= args.start and idx <= args.end:
            
        
if __name__ == "__main__":
    main ()