from torch.utils.data import DataLoader
from .gta5_dataset import *
from .cityscapes_dataset import *
from .idd_dataset import *

def make_datasets(opt):
    # source 
    if opt.source == 'GTA5':
        dataset_s_train = GTA5_dataset(opt, opt.source_list_path, opt.split_source_train)
    elif opt.source == 'Cityscapes':
        dataset_s_train = Cityscapes_dataset(opt, opt.source_list_path, opt.split_source_train)
    elif opt.source == 'IDD':
        dataset_s_train = IDD_dataset(opt, opt.source_list_path, opt.split_source_train)

    # target 
    if opt.target == 'GTA5':
        dataset_t_train = GTA5_dataset(opt, opt.target_list_path, opt.split_target_train)
        dataset_t_test = GTA5_dataset(opt, opt.target_list_path, opt.split_target_test)
    elif opt.target == 'Cityscapes':
        dataset_t_train = Cityscapes_dataset(opt, opt.target_list_path, opt.split_target_train)
        dataset_t_test = Cityscapes_dataset(opt, opt.target_list_path, opt.split_target_test)
    elif opt.target == 'IDD':
        dataset_t_train = IDD_dataset(opt, opt.target_list_path, opt.split_target_train)
        dataset_t_test = IDD_dataset(opt, opt.target_list_path, opt.split_target_test)

    return dataset_s_train, dataset_t_train, dataset_t_test
    # return source train, target train, target test

def make_dataloader(opt):
    dataset_s_train, dataset_t_train, dataset_t_test = make_datasets(opt)
    loader_s_train = DataLoader(dataset = dataset_s_train,
                                    batch_size = opt.batch_size,
                                    shuffle = opt.shuffle,
                                    drop_last = opt.drop_last)

    loader_t_train = DataLoader(dataset = dataset_t_train,
                                    batch_size = opt.batch_size,
                                    shuffle = opt.shuffle,
                                    drop_last = opt.drop_last)

    loader_t_test = DataLoader(dataset = dataset_t_test,
                                    batch_size = 1,
                                    shuffle = False,
                                    drop_last = False)

    return loader_s_train, loader_t_train, loader_t_test