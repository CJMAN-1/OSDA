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

        self.ignore_label = 255

    def __len__(self):
        return self.num_data

    def __getitem__(self):
        # 여기는 dataset마다 각각 다른 transform이 들어갈 수 있으므로 각 데이터셋에서 작성.
        pass

    def img_transform(self):
        pass

    def label_transform(self):
        pass

    def convert_id_to_trainid(self, label): # label: HW
        label_cvt = self.ignore_label*torch.ones_like(label)
        for id, tid in self.id_to_trainid.items():
            label_cvt[label == id] = tid # torch에서 == 연산 overloading했음
        return label_cvt
        
    def colorize_label(self, label): # label: BHW -> label_color:B3HW
        size = list(label.size())
        size.insert(1, 3)
        temp = torch.ones(size[2:]).cuda()
        label_color = torch.zeros(size).cuda()
        for b in range(size[0]):
            for c in range(3):
                for i, color in enumerate(self.colors):
                    temp = (label[b,:,:] == i) * color[c]
                    label_color[b,c,:,:] += temp

        return label_color