import os
import torch

from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

# GTA5 데이터셋을 기준으로 함.
class Custom_dataset(Dataset):
    def __init__(self, opt, listfile_path, split):
        self.opt = opt
        # data listfile path 지정
        imgs_listfile_path = os.path.join(listfile_path, split+'_imgs.txt')
        labels_listfile_path = os.path.join(listfile_path, split+'_labels.txt')

        # listfile에서 data path 불러오기
        with open(imgs_listfile_path, 'r') as f:
            self.imgs_list = f.read().splitlines()
        with open(labels_listfile_path, 'r') as f:
            self.labels_list = f.read().splitlines()
        self.num_data = len(self.imgs_list)

    def __len__(self):
        return self.num_data

    def __getitem__(self):
        # 여기는 dataset마다 각각 다른 transform이 들어갈 수 있으므로 각 데이터셋에서 작성.
        pass

    def img_transform(self):
        pass

    def label_transform(self):
        pass