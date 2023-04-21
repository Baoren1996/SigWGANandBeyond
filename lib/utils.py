import pickle
import json
import yaml
import numpy as np
import torch
from typing import Dict,Any, Iterable, Union




def sample_indices(dataset_size, batch_size):
    indices = torch.from_numpy(np.random.choice(dataset_size, size=batch_size, replace=False)).cuda().long()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def json_it(obj, filename):
    with open(filename, 'w') as f:
        json.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()

def train_test_split(data,seed=0,test_ratio=None,shuffle=True):
    """
    Split the input data into test set and training set

    Input:
      data: the dataset we want to split,
      seed: random seed 0 by default 
      test_ratio: the ratio of test size over size of dataset 
      shuffle: split the set by shuffling

    Output：
      list： a list of splited datset.
    """
    if test_ratio==None:
      test_ratio=0.25

    #compute train and test size
    n_samples=len(data)
    n_train=int(n_samples*(1-test_ratio))
    n_test=n_samples-n_train
    #get index for splitting
    if shuffle is False:
       train=np.arange(n_train)
       test=np.arange(n_train,n_train+n_test)
    else:
      rng=np.random.default_rng(seed)
      permutation = rng.permutation(n_samples)
      test = permutation[:n_test]
      train = permutation[n_test : (n_test + n_train)]
      
    return data[train],data[test]

def load_yaml_file(path,allow_unsafe=False):
    '''
    Load yaml config file
    Input
        path: the yaml file path
        allow_unsafe: allow unsafe load or not 
    Ouput 
        cfg: config dict 
    '''
    with open(path) as f:
        try:
            cfg=yaml.safe_load(f)
        except yaml.constructor.ConstructorError:
            if not allow_unsafe:
                raise Warning('Loading {} file using unsafe_load can be risky if it constains malicious content'.format(path))
            f.close()
            with open(path) as f:
                cfg = yaml.unsafe_load(f)
    return cfg 

def merge_a_into_b(a: dict, b) -> None:
         # merge dict a into dict b. values in a will overwrite b.
            for k, v in a.items():
                 b.update(k,v)

def save_cfg_to_yaml(cfg,path):
    files = open(path,'w')
    yaml.dump(cfg,files)
    files.close()



_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def grad_norm(parameters: _tensor_or_tensors, norm_type: float = 2.0, error_if_nonfinite: bool = False,) -> torch.Tensor:
    r"""Gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Note: currently only support parameters all on one single device.
    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
    Returns:
        Total norm of the parameter gradients (viewed as a single vector).
    """

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.)

    norms = [g.detach().data.norm(norm_type) for g in grads]
    total_norm = norms[0] if len(norms) == 1 else torch.norm(torch.stack(norms), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    return total_norm
    
