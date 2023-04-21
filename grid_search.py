from train import *
from sklearn.model_selection import ParameterGrid 
import pandas as pd 
from tqdm import tqdm 
import numpy as np
import copy
from lib.augmentations import parse_augmentations


#Grid of hyperparameters 
#param_grid = {
#      "batch_size": [100,256],
#      "hidden_dims": [[50,50,50],],
#      "p": [2,3,4],
#      "q": [2,3,4],
#      "G_lr": [2e-4,1e-4],
#      "D_lr": [2e-4,1e-4],
#      "G_beta1": [0.,],
#      "G_beta2": [0.9,],
#      "D_beta1": [0.,],
#      "D_beta2": [0.9,],
#      "total_steps": [1000,2000],
#      "num_D_steps": [2,3,4,],
#      "mc_size": [1000,],
#      }
param_grid = {
      "batch_size": [100],
      "hidden_dims": [[50,50,50],],
      "p": [3,4],
      "q": [3,4],
      "G_lr": [1e-2],
      "gamma": [0.9,],
      "opt_step_size": [100,],
      "total_steps": [1000,],
      }

#AUGMENTATIONS = { 'Cumsum': Cumsum, 'LeadLag': LeadLag, 'Scale': Scale, 'AddLags':  AddLags, 'Concat': Concat,'Addtime': Addtime}
aug_list=[[{"name":  "Scale", "scale":  x}, {"name":  "Cumsum"},{"name": "AddLags","m":2}, {"name": "LeadLag","with_time":True}] for x in [0.01,0.05,0.1,0.2,0.4,0.8]]
sig_grid = {
     "mc_size": [500,1000],
     "depth":[2,3],
     "basepoint":[True,False],
     "augmentations":aug_list,#[ [ {"name":  "Scale", "scale":  1}, {"name":  "Addtime"} ], [ {"name":  "Scale", "scale":  0.8}, {"name":  "Addtime"} ]],
    }

def run(algo_id, base_config,algo_config, base_dir, dataset, spec, data_params={},test_ratio=0.2):
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
    x_real_train, x_real_test = train_test_split(x_real,seed=base_config.seed,test_ratio =test_ratio)#x_real[:ind_train], x_real[ind_train:]

    #Get algorithm 
    algo = get_algo(algo_id, base_config, dataset, data_params, x_real_train,algo_config)
    # Train the algorithm
    algo.fit()
    return algo 


def get_algo(algo_id, base_config, dataset, data_params, x_real,algo_config=None):
    if algo_id in ['SigCWGAN']:
        algo = ALGOS[algo_id](x_real=x_real, config=algo_config, base_config=base_config)
    else:
        algo = ALGOS[algo_id](x_real=x_real, base_config=base_config)
    return algo


def grid_searchn(args,suffix=None):
    print('Start of GridSearch. CUDA: %s' % args.use_cuda)
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
                    #create dataframe to store the test metric for every hyperparameter set
                    df = pd.DataFrame()
                    i=0
                    if algo_id=='SigCWGAN':
                        param_grid.update(sig_grid)
                    # get the list of all hyperparameter set 
                    grid = list(ParameterGrid(param_grid))
                    num_parm = len(grid)
                    #go through all hyperparameter set in the grid 
                    for param in grid:
                        i+=1
                        print('{i}/{n} of Parameter Grid'.format(i=i, n=num_parm))
                        print(param)  
                        merge_a_into_b(param,base_config)
                        # if SigCWGAN, get the algo_config for signature configuration 
                        if algo_id=='SigCWGAN': 
                            augmentations = parse_augmentations(copy.deepcopy(param["augmentations"])) #use deep copy here
                            algo_config = SigCWGANConfig(
                                    mc_size=param["mc_size"],
                                    sig_config_past= SignatureConfig(depth=param["depth"], basepoint=param["basepoint"],augmentations=augmentations),
                                    sig_config_future=SignatureConfig(depth=param["depth"], basepoint=param["basepoint"],augmentations=augmentations),
                                    ) 
                        else:
                            algo_config=None

                        algo = run(
                                algo_id=algo_id,
                                base_config=base_config,
                                algo_config=algo_config,
                                data_params=data_params,
                                dataset=dataset,
                                base_dir=args.base_dir,
                                spec=spec, 
                                )
                        record = param.copy()
                        record['abs_metric'] = np.mean(algo.training_loss['abs_metric'][-10:])
                        record['acf_id'] = np.mean(algo.training_loss['acf_id'][-10:])
                        if algo.is_multivariate:
                            record['cross_correl'] = np.mean(algo.training_loss['cross_correl'][-10:])
                            record['best_score'] = algo.best_score
                        df = df.append(record, ignore_index=True) # use  pd.concate in the future 
                        df.to_csv(pt.join(pt.dirname(base_config.experiment_directory), '{algo}/grid_search_{suffix}.csv'.format(algo=algo_id,suffix=suffix)))



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser() 
    # Meta parameters
    parser.add_argument('-base_dir', default='./gridsearch_results', type=str)
    parser.add_argument('-use_cuda',  default=True)#action='store_true')
    parser.add_argument('-num_seeds', default=1, type=int)
    parser.add_argument('-initial_seed', default=0, type=int)
    parser.add_argument('-datasets', default=['STOCKS',], nargs="+")
    # algo list 'SigCWGAN','MCGAN','SigMCGAN','GMMN', 'RCGAN', 'TimeGAN', 'RCWGAN','ProSigMCGAN',
    parser.add_argument('-algos', default=['SigCWGAN',], nargs="+")
    args = parser.parse_args()
    grid_searchn(args)