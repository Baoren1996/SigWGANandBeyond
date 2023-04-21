import itertools
import os
from os import path as pt

import matplotlib.pyplot as plt
import numpy as np
import torch

from hyperparameters import *
from lib import ALGOS
from lib.algos.base import BaseConfig
from lib.data import download_man_ahl_dataset, download_mit_ecg_dataset
from lib.data import get_data
from lib.plot import savefig, create_summary
from lib.utils import pickle_it, train_test_split, load_yaml_file, merge_a_into_b,set_seed, save_cfg_to_yaml,json_it


def get_algo_config(dataset, data_params):
    """ Get the algorithms parameters. """
    key = dataset
    if dataset == 'VAR':
        key += str(data_params['dim'])
    elif dataset == 'STOCKS':
        key += '_' + '_'.join(data_params['assets'])
    return SIGCWGAN_CONFIGS[key]

def update_config_from_yaml(base_config,algo_id):
    config_path = f'configs/{algo_id}.yaml'
    cfg = load_yaml_file(config_path)
    merge_a_into_b(cfg,base_config)



def get_algo(algo_id, base_config, dataset, data_params, x_real):
    if algo_id in ['SigCWGAN','SigMCGAN']:
        algo_config = get_algo_config(dataset, data_params)
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo


def run(algo_id, base_config, base_dir, dataset, spec, data_params={},test_ratio=0.2):
    """ Create the experiment directory, calibrate algorithm, store relevant parameters. """
    print('Executing: %s, %s, %s' % (algo_id, dataset, spec))
    experiment_directory = pt.join(base_dir, dataset, spec, 'seed={}'.format(base_config.seed), algo_id)
    if not pt.exists(experiment_directory):
        # if the experiment directory does not exist we create the directory
        os.makedirs(experiment_directory)
    #update experiment directory in config
    base_config.experiment_directory=experiment_directory
    
    # Set seed for exact reproducibility of the experiments
    set_seed(base_config.seed)
    # initialise dataset and algo
    x_real = get_data(dataset, base_config.p, base_config.q, **data_params)

    x_real = x_real.to(base_config.device) 
    
    #Train Test Split
    #ind_train = int(x_real.shape[0] * train_ratio) 
    x_real_train, x_real_test = train_test_split(x_real,seed=base_config.seed,test_ratio =test_ratio)#x_real[:ind_train], x_real[ind_train:]
    print('Training size: ',x_real_train.shape)
    print('Test size: ',x_real_test.shape)
    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train)
    # Train the algorithm
    algo.fit()
    # create summary
    print('Saving model weights and plot')
    create_summary(dataset, base_config.device, algo.G, base_config.p, base_config.q, x_real_test)
    savefig('summary.png', experiment_directory)
    x_fake = create_summary(dataset, base_config.device, algo.G, base_config.p, 8000, x_real_test, one=True)
    savefig('summary_long.png', experiment_directory)
    plt.plot(x_fake.cpu().numpy()[0, :2000])
    savefig('long_path.png', experiment_directory)
    # Pickle generator weights, real path and hyperparameters.
    pickle_it(x_real_test.to('cpu'), pt.join(pt.dirname(experiment_directory), 'x_real_test.torch'))
    pickle_it(x_real_train.to('cpu'), pt.join(pt.dirname(experiment_directory), 'x_real_train.torch'))
    pickle_it(x_real.to('cpu'), pt.join(pt.dirname(experiment_directory), 'x_real.torch'))
    pickle_it(algo.training_loss, pt.join(experiment_directory, 'training_loss.pkl'))
    pickle_it(algo.G.to('cpu').state_dict(), pt.join(experiment_directory, 'G_weights.torch'))
    #save config as yaml 
    save_cfg_to_yaml(base_config,pt.join(experiment_directory, 'base_config.yaml'))
    #save tne detail of the best score
    json_it(algo.best_list,pt.join(experiment_directory, 'best_score.json'))
    # Log some results at the end of training
    algo.plot_losses()
    savefig('losses.png', experiment_directory)
    print('All saved!')
    return algo 


def get_dataset_configuration(dataset):
    if dataset == 'ECG':
        generator = [('id=100', dict(filenames=['100']))]
    elif dataset == 'STOCKS':
        generator = (('_'.join(asset), dict(assets=asset)) for asset in [('SPX',),('SPX','DJI')])#,'TimeGAN'('SPX','DJI')
    elif dataset == 'VAR':
        par1 = itertools.product([1], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8)])
        par2 = itertools.product([2], [(0.2, 0.8), (0.5, 0.8), (0.8, 0.8), (0.8, 0.2), (0.8, 0.5)])
        par3 = itertools.product([3], [(0.2, 0.8), (0.8, 0.2), (0.8, 0.5),(0.5, 0.8), (0.8, 0.8),])
        par4 = itertools.product([10], [(0.8, 0.8),])
        par5 = itertools.product([50], [(0.8, 0.8),])
        combinations = itertools.chain(par3)
        generator = (
            ('dim={}_phi={}_sigma={}'.format(dim, phi, sigma), dict(dim=dim, phi=phi, sigma=sigma))
            for dim, (phi, sigma) in combinations
        )
    elif dataset == 'ARCH':
        generator = (('lag={}'.format(lag), dict(lag=lag)) for lag in [2,3,4])
    elif dataset == 'SINE':
        generator = [('a', dict())]
    else:
        raise Exception('%s not a valid data type.' % dataset)
    return generator


def main(args):
    if not pt.exists('./data'):
        os.mkdir('./data')
    #if not pt.exists('./data/oxfordmanrealizedvolatilityindices.csv'):
    #    print('Downloading Oxford MAN AHL realised library...')
    #    download_man_ahl_dataset()
    #if not pt.exists('./data/mitdb'):
    #    print('Downloading MIT-ECG database...')
    #    download_mit_ecg_dataset()
    print('Start of training. CUDA: %s' % args.use_cuda)
    for dataset in args.datasets:
        for algo_id in args.algos:
            for seed in range(args.initial_seed, args.initial_seed + args.num_seeds):
                base_config = BaseConfig()
                #load config from yaml file 
                update_config_from_yaml(base_config,algo_id)
                base_config.seed = seed
                #get dataset configuration
                generator = get_dataset_configuration(dataset)
                for spec, data_params in generator:
                    run(
                        algo_id=algo_id,
                        base_config=base_config,
                        data_params=data_params,
                        dataset=dataset,
                        base_dir=args.base_dir,
                        spec=spec, 
                    )

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser() 

    # Meta parameters
    parser.add_argument('-base_dir', default='./numerical_results', type=str)
    parser.add_argument('-use_cuda',  action='store_true')
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)
    parser.add_argument('-datasets', default=['STOCKS','ARCH','VAR'], nargs="+")

    # algo list 'SigCWGAN','MCGAN','SigMCGAN','GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN'
    parser.add_argument('-algos', default=[ 'SigCWGAN','MCGAN','SigMCGAN','GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN'], nargs="+")#'SigCWGAN',

    args = parser.parse_args()
    main(args)
