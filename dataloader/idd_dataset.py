from .custom_dataset import *
from torchvision.transforms.functional import InterpolationMode

class IDD_dataset(Custom_dataset):
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
        [255, 255, 255],     # unknown
        [0, 0, 0],          # ignore
        ]

        # label configure
        ignore_label = -1
        unknown_label = 19
        self.ignore_label = ignore_label
        self.unknown_label = unknown_label
        self.id_to_trainid = {
            #road                   #parking            #drivable fallback      #sidewalk               #rail track
            0:0,                    1:ignore_label,     2:unknown_label,        3:1,                    4:ignore_label,
            #non-drivable fallback  #person             #animal                 #rider                  #motorcycle
            5:9,                    6:11,               7:ignore_label,         8:12,                   9:17,
            #bicycle                #autorickshaw       #car                    #truck                  #bus
            10:18,                  11:unknown_label,   12:13,                  13:14,                  14:15,
            #caravan                #trailer            #train                  #vehicle fallback       #curb
            15:ignore_label,        16:ignore_label,    17:16,                  18:unknown_label,       19:unknown_label,
            #wall                   #fence              #guard rail             #billboard              #traffic sign
            20:3,                   21:4,               22:ignore_label,        23:unknown_label,       24:7,
            #traffic light          #pole               #pole group             #obs-str-bar-fallback   #building
            25:6,                   26:5,               27:ignore_label,        28:ignore_label,        29:2,
            #bridge                 #tunnel             #vegetation             #sky                    #fallback background
            30:ignore_label,        31:ignore_label,    32:8,                   33:10,                  34:-2,
            #unlabeled              #ego vehicle        #rectification border   #out of roi             #license plate
            35:ignore_label,        36:ignore_label,    37:ignore_label,        38:ignore_label,        39:ignore_label
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
            

        