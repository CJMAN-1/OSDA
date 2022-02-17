from .custom_dataset import *
from torchvision.transforms.functional import InterpolationMode

class Cityscapes_dataset(Custom_dataset):
    def __init__(self, opt, listfile_path, split):
        super().__init__(opt, listfile_path, split)

        # transform 
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Resize(size=self.opt.img_size, interpolation=InterpolationMode.BILINEAR),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.label_transform = transforms.Compose([
            transforms.PILToTensor(),
            #transforms.Resize(size=self.opt.img_size, interpolation=InterpolationMode.NEAREST),
        ])

        # colors
        self.colors = [
        [128, 64, 128],     # road
        [244, 35, 232],     # sidewalk
        [70, 70, 70],       # building
        [102, 102, 156],    # wall
        [190, 153, 153],    # fence
        [153, 153, 153],    # pole
        [250, 170, 30],     # traffic light
        [220, 220, 0],      # traffic sign
        [107, 142, 35],     # vegetation
        [152, 251, 152],    # terrain
        [0, 130, 180],      # sky
        [220, 20, 60],      # person
        [255, 0, 0],        # rider
        [0, 0, 142],        # car
        [0, 0, 70],         # truck
        [0, 60, 100],       # bus
        [0, 80, 100],       # train
        [0, 0, 230],        # motorcycle
        [119, 11, 32],      # bicycle
        [0, 0, 0]           # ignore
        ]

        # label configure
        ignore_label = -1
        self.ignore_label = ignore_label
        self.id_to_trainid = {
            #unlabeled          #egovehicle         #rectfication border#out of roi         #static
            0:ignore_label,     1:ignore_label,     2:ignore_label,     3:ignore_label,     4:ignore_label,
            #dynamic            #ground             #road               #sidewalk           #parking
            5:ignore_label,     6:ignore_label,     7:0,                8:1,                9:ignore_label,
            #rail track         #building           #wall               #fence              #guard rail
            10:ignore_label,    11:2,               12:3,               13:4,               14:ignore_label,
            #bridge             #tunnel             #pole               #polegroup          #traffic light
            15:ignore_label,    16:ignore_label,    17:5,               18:ignore_label,    19:6,
            #traffic sign       #vegetation         #terrain            #sky                #person
            20:7,               21:8,               22:9,               23:10,              24:11,
            #rider              #car                #truck              #bus                #caravan
            25:12,              26:13,              27:14,              28:15,              29:ignore_label,
            #trailer            #train              #motorcycle         #bicycle            #license plate
            30:ignore_label,    31:16,              32:17,              33:18,              -1:ignore_label,
        }
        self.validclass_name = [
            'road',             'sidewalk',         'building',         'wall',             'fence',
            'pole',             'traffic light',    'traffic sign',     'vegetation',       'terrain',
            'sky',              'person',           'rider',            'car',              'truck',
            'bus',              'train',            'motorcycle',       'bicycle'
        ]
        self.num_class = len(self.validclass_name)
        
    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        # load and transform
        img = Image.open(self.imgs_list[idx]).convert('RGB')
        label = Image.open(self.labels_list[idx])
        print(self.labels_list[idx])

        img = self.img_preprocess(img)
        label = self.label_preprocess(label)
        
        return img, label

    def img_preprocess(self, img):
        img = self.img_transform(img)
        return img

    def label_preprocess(self, label):
        label = self.label_transform(label)
        
        label = self.convert_id_to_trainid(label)
        return label

    def convert_id_to_trainid(self, label): # label: 1HW
        label_cvt = self.ignore_label*torch.ones_like(label)
        for id, tid in self.id_to_trainid.items():
            label_cvt[label == id] = tid # torch에서 == 연산 overloading했음
        return label_cvt
        
    def colorize_label(self, label): # label: B1HW
        size = list(label.size())
        temp = torch.ones(size[2:]).cuda()
        size[1] = 3
        label_color = torch.zeros(size).cuda()
        for b in range(size[0]):
            for c in range(3):
                for i, color in enumerate(self.colors):
                    temp = (label[b,0,:,:] == i) * color[c]
                    label_color[b,c,:,:] += temp

        return label_color
            

        