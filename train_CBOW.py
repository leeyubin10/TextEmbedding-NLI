# -*- coding: utf-8 -*-

import argparse, random, time
import numpy as np
import torch

from trainers import CBOWTrainer

def set_seed(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

class Dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def main():
    start = time.time()
    ############################################## EDIT ################################################
    config = {
        'seed': 42,
        'num_epochs': 50,
        'train_batch_size': 512,
        'val_batch_size': 512,
        'test_batch_size': 512,
        'embedding_dim' : 20,
        'max_length': 10,
        'lr': 5e-4,
        'eps':1e-4,
        'weight_decay': 5e-4,
        'save_name': 'CBOW_embedding',
    ############################################## EDIT ################################################
        'window_size' : 2
    }
    config = Dotdict(config)
    set_seed(config)

    trainer = CBOWTrainer(config)  # pass the name of the model to use
    trainer.train_and_test()

    print("Execution time : {:.4f} sec".format(time.time() - start))
    
if __name__ == '__main__':
    main()
