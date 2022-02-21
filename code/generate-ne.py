"""
    Generate Negation features from Salud Jiménez-Zafra
    
    @author José Antonio García-Díaz <joseantonio.garcia8@um.es>
    @author Rafael Valencia-Garcia <valencia@um.es>
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from dlsdatasets.DatasetResolver import DatasetResolver
from utils.Parser import DefaultParser


def main ():

    # var parser
    parser = DefaultParser (description = 'To generate negation features')
    
    
    # @var args Get arguments
    args = parser.parse_args ()
    
    
    # @var dataset_resolver DatasetResolver
    dataset_resolver = DatasetResolver ()
    
    
    # @var dataset Dataset This is the custom dataset for evaluation purposes
    dataset = dataset_resolver.get (args.dataset, args.corpus, args.task, False)
    dataset.filename = dataset.get_working_dir (args.task, 'dataset.csv')


    # @var df Ensure if we already had the data processed
    df = dataset.get ()
    
    
    # @var negations List
    negations = [
        'a_no_ser_que',
        'absolutamente-nada',
        'absolutamente-nadie',
        'absolutamente-no',
        'apenas',
        'apenas-nada',
        'aun-no',
        'aun-no-nada',
        'aún-nadie',
        'aún-no',
        'casi-apenas',
        'casi-nada',
        'casi-nadie',
        'casi-ningún',
        'casi-nunca',
        'cero',
        'excepto',
        'falta',
        'falta-de',
        'jamas',
        'jamas-jamas',
        'jamás',
        'jamás-jamás',
        'nada',
        'nada-aun',
        'nada-de',
        'nada-de-nada',
        'nada-mas',
        'nada-más',
        'nada-ni',
        'nada-ni-nadie',
        'nada-ni-por-nadie',
        'nada-no',
        'nada-no-nada',
        'nada-para',
        'nada-por',
        'nada-siempre',
        'nada-su',
        'nada-tan',
        'nadie',
        'nadie-en',
        'nadie-en-nada',
        'nadie-lo',
        'nadie-ni',
        'nadie-no',
        'nadie-para',
        'nadie-por-nada',
        'ni',
        'ni-apenas',
        'ni-de',
        'ni-de-nada',
        'ni-idea',
        'ni-idea-de',
        'ni-mas-ni',
        'ni-nada',
        'ni-nada-muy',
        'ni-nadie',
        'ni-ni',
        'ni-ninguno',
        'ni-no',
        'ni-nunca',
        'ni-para',
        'ni-siquiera',
        'ni-siquiera-de',
        'ni-su',
        'ni-tampoco',
        'ni-tan',
        'ningun',
        'ninguna',
        'ninguna-ni',
        'ninguno',
        'ninguno-de',
        'ningún',
        'no',
        'no-demasiado',
        'no-fue',
        'no-fue-de',
        'no-fue-ninguna',
        'no-lo',
        'no-mucho',
        'no-nada',
        'no-nadie',
        'no-ni',
        'no-ni-nunca',
        'no-ninguna',
        'no-ninguno',
        'no-ningún',
        'no-no',
        'no-no-nada',
        'no-por',
        'no-por-nada',
        'no_sólo',
        'nos',
        'nos-no',
        'nunca',
        'nunca-jamás',
        'nunca-nadie',
        'nunca-ni',
        'nunca-no',
        'nunca-no-nada',
        'sin',
        'sin-de',
        'sin-mucho',
        'sin-nada',
        'sin-nadie',
        'sin-ni',
        'sin-ningun',
        'sin-ninguna',
        'sin-ningún',
        'sin-no',
        'sin-para',
        'sin-tanto',
        'sin_necesidad_de',
        'tampoco',
        'tampoco-nadie',
        'tampoco-para-la',
        'tampoco-tan',
        'todavia',
        'todavia-no',
        'ya-nada',
        'ya-ni',
        'ya-no'
    ]
    
    
    # @var df_negations_list Series
    df_negations_list = pd.Series (negations)
    
    
    # Open files
    df_neg = pd.concat ([
        pd.read_csv (dataset.get_working_dir ('dataset', 'neg.csv'))
    ], axis = 0)[['total_negation_cues', 'negation_cues_list']]
    
    
    # Fill
    for negation_clue in df_negations_list.iteritems ():
        df_neg['neg_' + str(negation_clue[1])] = df_neg['negation_cues_list'].str.count (r"\b" + str (negation_clue[1]) + r"\b")
    df_neg = df_neg.fillna (0.0)
    df_neg = df_neg.drop (['negation_cues_list'], axis = 1)

    
    # Save features
    df_neg.to_csv (dataset.get_working_dir ('ne.csv'), index = False)
    
    
    
    
if __name__ == "__main__":
    main ()
