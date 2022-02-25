from trainer.trainer import Trainer
from param import *
import torch
import numpy
import random

if __name__ == '__main__':
    opt = get_params()

    torch.manual_seed(opt.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    numpy.random.seed(opt.random_seed)
    random.seed(opt.random_seed)

    trainer = Trainer(opt)
    trainer.train()
    