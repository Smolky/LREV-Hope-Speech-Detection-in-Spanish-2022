"""
    To perform feature selection
    
    Right now this script only applies MinMaxScaler and a feature selection
    based on IG. These techniques were applied first for the LF; however, 
    we make another tests to another types of features. Note that we do not 
    apply this feature selection on the tokenizer (we)
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn import preprocessing
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD

from tqdm import tqdm
from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser
from features.FeatureResolver import FeatureResolver


def main ():

    # var parser
    parser = DefaultParser (description = 'Feature selection')
    
    
    # @var selectable_features List
    selectable_features = ['lf', 'be', 'se', 'ne', 'cf', 'bf', 'pr', 'ng', 'it']
    
    
    # Add parser
    parser.add_argument ('--features', 
        dest = 'features', 
        default = 'all', 
        help = 'Select the family or features to select', 
        choices = ['all'] + selectable_features
    )
    parser.add_argument ('--n', 
        dest = 'n', 
        default = 5, 
        help = 'Number of components for LSA'
    )
    
    
    # @var feature_folder String
    feature_folder = 'features'
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    
    
    # Determine if we need to use the merged dataset or not
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Dataframe
    df = dataset.get ()
    df_train = df.loc[df["__split"] == "train"]

    feature_resolver = FeatureResolver(dataset)
    transformer = feature_resolver.get("it", dataset.get_working_dir(args.task, "it.csv"))
    feature_df = transformer.transform([])
    feature_df_train = feature_df.loc[df_train.index]

    # SVD
    n = int(args.n)
    svd = TruncatedSVD(
        n_components = n,
        n_iter       = 5,
        random_state = 0
    )
    
    # Training
    svd.fit(feature_df_train)

    # All
    Y = svd.transform(feature_df)
    Y = pd.DataFrame(Y)
    Y.columns = ['it_' + str (x) for x in range(n)]
    Y.to_csv(dataset.get_working_dir(args.task, "it_lsa_" + str(n) + ".csv"), index = False)

if __name__ == "__main__":
    main ()
