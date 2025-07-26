import pandas as pd
import numpy as np
from preprocessing import *
from functools import partial

def get_scores(cases, kan_hyperparams, fnn_hyperparams, preprocess, trials=30):
    # set up data structure to store values
    kan = {}
    fnn = {}
    # loop through each dataset 30 times
    for case in cases:
        kan[case] = {}
        fnn[case] = {}
        for test in range(30):
            # preprocess dataset
            dataset = preprocess[case](shuffle=True)
            # fit a kan with best hyperparams and training set

            # predict test set results with kan

            # fit an fnn with best hyperparams and training set

            # predict test set results with fnn

            outputs = dataset['output_labels']
            for output in outputs:
                kan[case][output] = {'MAE': [], 'R2': []}
                fnn[case][output] = {'MAE': [], 'R2': []}

                # get mae for kan 
                kan[case][output]['MAE'].append( ? )
                # get mae for fnn
                fnn[case][output]['MAE'].append( ? )
                # get r2 for kan
                kan[case][output]['R2'].append( ? )
                # get r2 for fnn
                fnn[case][output]['R2'].append( ? )

    return kan, fnn

def get_wilcoxon(kan_dict, fnn_dict):
    return

if __name__ == "__main__":
    cases = snakemake.params.d_list
    preprocessing_funcs = {
        'CHF': get_chf,
        'BWR': get_bwr,
        'MITR_A': partial(get_mitr, region='A'),
        'MITR_B': partial(get_mitr, region='B'),
        'MITR_C': partial(get_mitr, region='C'),
        'MITR': get_mitr,
        'XS': get_xs,
        'FP': get_fp,
        'HEAT': get_heat,
        'REA': get_rea,
        'HTGR': get_htgr
    }
    kan_hyperparams =  {
    'FP': {
         'depth': 1, 'grid': 8, 'k': 7, 'lamb': 2.0426962767412815e-05, 'lamb_entropy': 5.0346373804560525, 'lr_1': 1.5, 'lr_2': 1.75, 'reg_metric': 'edge_forward_sum', 'steps': 75
         },
    'BWR': {
         'depth': 1, 'grid': 7, 'k': 2, 'lamb': 0.0008912210456241697, 'lamb_entropy': 7.488094627223641, 'lr_1': 1.75, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 125
         },
    'HEAT': {
         'depth': 1, 'grid': 7, 'k': 3, 'lamb': 0.00018986520634595234, 'lamb_entropy': 8.209205342922996, 'lr_1': 1.5, 'lr_2': 2, 'reg_metric': 'edge_forward_spline_u', 'steps': 150
         },
    'HTGR': {
         'depth': 1, 'grid': 8, 'k': 3, 'lamb': 1.2166058649376336e-05, 'lamb_entropy': 7.665809430995193, 'lr_1': 0.75, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 25
         },
    'MITR_A': {
         'depth': 1, 'grid': 4, 'k': 7, 'lamb': 4.2843417969236176e-05, 'lamb_entropy': 0.025403293360453105, 'lr_1': 1.5, 'lr_2': 1.25, 'reg_metric': 'edge_forward_sum', 'steps': 25
         },
    'MITR_B' : {
         'depth': 1, 'grid': 3, 'k': 2, 'lamb': 1.1115022163426145e-06, 'lamb_entropy': 0.9626840845416629, 'lr_1': 0.75, 'lr_2': 1, 'reg_metric': 'edge_forward_sum', 'steps': 250
         },
    'MITR_C': {
         'depth': 1, 'grid': 3, 'k': 3, 'lamb': 9.739068562318698e-06, 'lamb_entropy': 7.047425350169997, 'lr_1': 1, 'lr_2': 1.5, 'reg_metric': 'edge_forward_spline_n', 'steps': 150
         },
    'MITR': {
         'depth': 1, 'grid': 7, 'k': 6, 'lamb': 0.00033697982852750485, 'lamb_entropy': 5.7732177455173055, 'lr_1': 1.75, 'lr_2': 0.5, 'reg_metric': 'edge_forward_spline_u', 'steps': 150
         },
    'CHF': {
         'depth': 1, 'grid': 9, 'k': 2, 'lamb': 5.123066656699474e-06, 'lamb_entropy': 7.716618914050463, 'lr_1': 2, 'lr_2': 1.75, 'reg_metric': 'edge_forward_spline_u', 'steps': 100
         },
    'REA': {
         'depth': 1, 'grid': 6, 'k': 8, 'lamb': 3.79496703629217e-05, 'lamb_entropy': 0.006504868427044119, 'lr_1': 2, 'lr_2': 0.5, 'reg_metric': 'edge_forward_sum', 'steps': 25
         },
    'XS': {
         'depth': 2, 'grid': 9, 'k': 4, 'lamb': 0.00039029273996368227, 'lamb_entropy': 0.42860645226254324, 'lr_1': 1.25, 'lr_2': 1.25, 'reg_metric': 'edge_forward_spline_u', 'steps': 100
         },
}
    fnn_hyperparams = {
        'CHF': {
            'hidden_nodes' : [231, 138, 267],
            'num_epochs' : 200,
            'batch_size' : 64,
            'learning_rate' : 0.0009311391232267503,
            'use_dropout': True,
            'dropout_prob': 0.4995897609454529,
        },
        'BWR': {
            'hidden_nodes' : [511, 367, 563, 441, 162],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0009660778027367906,
            'use_dropout': False,
            'dropout_prob': 0,
        },
        'FP': {
            'hidden_nodes' : [66, 400],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.001,
            'use_dropout': False,
            'dropout_prob': 0,
        },
        'HEAT': {
            'hidden_nodes' : [251, 184, 47],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008821712781015931,
            'use_dropout': False,
            'dropout_prob': 0,
        },
        'HTGR': {
            'hidden_nodes' : [199, 400],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.00011376283985074373,
            'use_dropout': True,
            'dropout_prob': 0.3225718287912892,
        },
        'MITR': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,           
        },
        'REA': {
            'hidden_nodes' : [326, 127],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0009444837105276597,
            'use_dropout': False,
            'dropout_prob': 0,         
        },
        'XS': {
            'hidden_nodes' : [95],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0003421585453407753,
            'use_dropout': False,
            'dropout_prob': 0,        
        },
        'MITR_A': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,        
        },
        'MITR_B': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,           
        },
        'MITR_C': {
            'hidden_nodes' : [309],
            'num_epochs' : 200,
            'batch_size' : 8,
            'learning_rate' : 0.0008321972582830564,
            'use_dropout': False,
            'dropout_prob': 0,        
        },        
    }
    kan_dict, fnn_dict = get_scores(
                            cases = snakemake.input.d_list,
                            kan_hyperparams = kan_hyperparams, 
                            fnn_hyperparams = fnn_hyperparams,
                            preprocess = preprocessing_funcs,
                            trials=snakemake.params.trials)
    get_wilcoxon(
        kan_dict = kan_dict,
        fnn_dict = fnn_dict,
        save_as = snakemake.output.wilcoxon_scores)
    