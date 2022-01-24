from custom_dataset import *

# cuda 관련한거는 trainer에서 batch가져올 때 변환.
class GTA5_dataset(Custom_dataset):
    def __init__(self, opt):
        # list_path, split_type, crop_size등을 opt에 담아서 전달.
        self.num_data = 0
        pass

    def __len__(self):
        return self.num_data

    def __getitem__(self, idx):
        # transform and 
        pass