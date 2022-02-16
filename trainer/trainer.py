from time import sleep
from dataloader.util_datasets import *
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        # dataloader
        self.loader_s_train, self.loader_t_train, self.loader_t_test = make_dataloader(opt)

        # tensorboard
        self.writer = SummaryWriter(log_dir=self.opt.logdir)

    def get_batch(self, loader_s, loader_t):
        batch = dict()
        data = dict()
        try:
            data['img'], data['label'] = loader_s.next()
            batch['S'] = data
        except StopIteration:
            loader_s = iter(self.loader_s_train)
            data['img'], data['label'] = loader_s.next()
            batch['S'] = data
        
        try:
            data['img'], data['label'] = loader_t.next()
            batch['T'] = data
        except StopIteration:
            loader_t = iter(self.loader_t_train)
            data['img'], data['label'] = loader_t.next()
            batch['T'] = data

        batch['S']['img'] = batch['S']['img'].cuda()
        batch['S']['label'] = batch['S']['label'].cuda()
        batch['T']['img'] = batch['T']['img'].cuda()
        batch['T']['label'] = batch['T']['label'].cuda()
        return batch

    def train(self):
        loader_s, loader_t = iter(self.loader_s_train), iter(self.loader_t_train)

        for iteration in range(1):#range(self.opt.total_iter):
            batch = self.get_batch(loader_s, loader_t)

            print(batch['S']['img'].size())
            print(batch['S']['label'].size())
            print(batch['T']['img'].size())
            print(batch['T']['label'].size())
            print(torch.max(batch['S']['img']))
            print(torch.max(batch['S']['label']))
            print(torch.min(batch['S']['img']))
            print(torch.min(batch['S']['label']))
            lbl = loader_s._dataset.colorize_label(batch['S']['label'])            

            img_grid = torchvision.utils.make_grid(batch['S']['img'], normalize=True, value_range=(-1,1))
            self.writer.add_image('input/img', img_grid)
            img_grid = torchvision.utils.make_grid(lbl, normalize=True, value_range=(0,255))
            self.writer.add_image('input/label', img_grid)

            sleep(0.5) # 아마 이미지가 log파일에 저장되는데 어느정도 시간이 필요한 것 같음.
