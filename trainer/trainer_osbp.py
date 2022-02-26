from torch.utils.tensorboard import SummaryWriter
from time import sleep
import sys
from dataloader.util_datasets import *
import torch
import torchvision
from networks.deeplabv2 import *
from util import metric
from prettytable import PrettyTable
from tqdm import tqdm
from torch import optim
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from torchsummaryX import summary
import pandas as pd

# openset domain adaptation by backpropagation (ECCV 2018)
class Trainer:
    def __init__(self, opt):
        self.opt = opt
        # dataloader
        self.loader = dict()
        self.loader['S_t'], self.loader['T_t'], self.loader['T_v'] = make_dataloader(opt)
        
        self.num_cls = dict()
        self.num_cls['S'] = self.loader['S_t'].dataset.num_class
        self.num_cls['T'] = self.loader['T_t'].dataset.num_class
        
        # segmentation model
        self.model = {}
        self.model['Task'] = Deeplab(num_classes= 19, initialization=self.opt.init_imgnet).cuda()

        # optimizer
        self.optimizer = {}
        self.optimizer['Task'] = optim.SGD(self.model['Task'].parameters(),
                                        lr=self.opt.lr,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weight_decay)

        
        # learning rate scheduler
        self.scheduler = {}
        self.scheduler['Task'] = optim.lr_scheduler.LambdaLR(optimizer=self.optimizer['Task'],
                                        lr_lambda=lambda epoch: 0.1 ** epoch,
                                        last_epoch=-1,
                                        verbose=False)
        
        # tensorboard
        self.writer = SummaryWriter(log_dir=self.opt.log_dir, filename_suffix=self.opt.ex)

        # log
        self.LOG = logging.getLogger(self.opt.ex)
        file_handler = logging.FileHandler(filename=os.path.join(self.opt.log_dir,'info.log'))
        formatter = logging.Formatter()
        file_handler.setFormatter(formatter)
        self.LOG.addHandler(file_handler)
        self.LOG.setLevel(logging.INFO)

        self.best_miou = 0
            # hyperparameter logging
        self.LOG.info('='*20 + 'PARAMETER' + '='*20)
        for k, v in self.opt._get_kwargs():
            self.LOG.info(f"{k:20} : {v}")

            # model logging
        self.LOG.info('='*20 + 'MODEL' + '='*20)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('expand_frame_repr', False)
        df_model = summary(self.model['Task'], torch.zeros(4,3,512,1024).cuda(), torch.zeros(4,512,1024).long().cuda())
        self.LOG.info(df_model.replace(np.nan, "-"))
        self.LOG.info('='*50)

        # TBD


    def get_batch(self, loader): # -> img : B3HW, label : BHW
        batch = dict()
        data = dict()
        for k in loader.keys():
            try:
                data['img'], data['label'] = loader[k].next()
                batch[k[0]] = data.copy()
            except StopIteration:
                loader[k] = iter(self.loader[k])
                data['img'], data['label'] = loader[k].next()
                batch[k[0]] = data.copy()

        batch['S']['img'] = batch['S']['img'].cuda()
        batch['S']['label'] = batch['S']['label'].cuda()
        batch['T']['img'] = batch['T']['img'].cuda()
        batch['T']['label'] = batch['T']['label'].cuda()
        return batch

    def train(self):
        loader = dict()
        loader['S_t'], loader['T_t'] = iter(self.loader['S_t']), iter(self.loader['T_t'])

        with logging_redirect_tqdm():
            for iteration in tqdm(range(1, self.opt.total_iter+1), desc='Training'):
                # get batch
                batch = self.get_batch(loader)

                # train model
                self.model['Task'].train()
                self.model['Task'].zero_grad()
                loss_task, pred_lbl = self.model['Task'](batch['S']['img'], batch['S']['label'])
                loss_task.backward()

                # updata model
                self.optimizer['Task'].step()
                self.LOG.info(f'iter : {iteration} | loss : {loss_task:.2f}')
                if iteration % self.opt.lr_schedule_freq ==0:
                    pre = self.scheduler['Task'].get_lr()
                    self.scheduler['Task'].step()
                    cur = self.scheduler['Task'].get_lr()
                    self.LOG.info(f'learning rate chnage {pre} -> {cur}')
                
                # tensorboard
                if iteration % self.opt.tensor_freq == 0:
                    with torch.no_grad():
                        img_grid = torchvision.utils.make_grid(batch['S']['img'], normalize=True, value_range=(-1,1))
                        self.writer.add_image('input/img', img_grid, iteration)
                        lbl = self.loader['S_t'].dataset.colorize_label(batch['S']['label'])
                        img_grid = torchvision.utils.make_grid(lbl, normalize=True, value_range=(0,255))
                        self.writer.add_image('input/GT', img_grid, iteration)
                        lbl = self.loader['S_t'].dataset.colorize_label(pred_lbl)
                        img_grid = torchvision.utils.make_grid(lbl, normalize=True, value_range=(0,255))
                        self.writer.add_image('output/prediction', img_grid, iteration)
                        sleep(0.5) # 아마 이미지가 log파일에 저장되는데 어느정도 시간이 필요한 것 같음.

                # evaluation
                if iteration % self.opt.eval_freq == 0:
                    self.eval() 
        
        print(f'학습끝 best miou : {self.best_miou:.2f}')

    def eval(self):
        self.model['Task'].eval()
        conf_mat = np.zeros((self.num_cls['T'],) * 2)
        miou = np.zeros(shape=1)
        iou = np.zeros(shape=self.num_cls['T']+1)
        
        for img, lbl in tqdm(self.loader['T_v'], leave=False, desc='Evaluation'):
            img = img.cuda()
            lbl = lbl.cuda()

            pred_lbl = self.model['Task'](img, lbl)

            # IoU 계산
            conf_mat += metric.conf_mat(lbl.cpu().numpy(), pred_lbl.cpu().numpy(), self.num_cls['T'])

        iou = 100 * np.diag(conf_mat)/ (np.sum(conf_mat, axis=0) + np.sum(conf_mat, axis=1) - np.diag(conf_mat))
        miou = np.nanmean(iou)
        if miou > self.best_miou:
            self.best_miou = miou
            self.LOG.info(f'best miou : {self.best_miou:.2f}')
            torch.save(self.model['Task'].state_dict(), os.path.join(self.opt.log_dir, 'model.pth'))

        # mIoU table 출력
        with logging_redirect_tqdm():
            table = PrettyTable()
            table.field_names = self.loader['T_v'].dataset.validclass_name[0:10]
            table.add_row(np.round(iou, 2)[0:10])
            self.LOG.info(table.get_string())
            table.clear_rows()
            table.field_names = self.loader['T_v'].dataset.validclass_name[10:] + ['mIoU']
            table.add_row(np.concatenate((np.round(iou,2)[10:], [np.round(miou,2)])))
            self.LOG.info(table.get_string())
        