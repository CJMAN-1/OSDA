from argparse import *
import argparse
import os

def get_params():
    # 파라미터를 받는 인스턴스 생성
    parser = argparse.ArgumentParser()

    # 데이터셋 관련
        # source
    parser.add_argument('--source', default='gta5', help='source dataset')
    parser.add_argument('--source_list_path', default='./datasets/GTA5/list', help='source data .txt file path')
    parser.add_argument('--split_source_train', default='all', help='source train data | all | train | test | val')

        # target
    parser.add_argument('--target', default='gta5', help='target dataset')
    parser.add_argument('--target_list_path', default='./datasets/GTA5/list', help='target data .txt file path')
    parser.add_argument('--split_target_train', default='train', help='target train data | all | train | test | val') 
    parser.add_argument('--split_target_test', default='val', help='target test data | all | train | test | val') 

        # common
    parser.add_argument('--img_size', default=(512,1024), help='(h,w), image size for train and test')


    # 학습에 관련된 하이퍼 파라미터
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--shuffle', default=False, help='shuffle dataloader')
    parser.add_argument('--drop_last', default=True, help='drop last batch')
    parser.add_argument('--total_iter', default=10000, help='total iteration')
    parser.add_argument('--lr', default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', default=0.001, help='learning rate')

    # 환경설정 파라미터
    parser.add_argument('--random_seed', default=3621, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, help='use cuda')

    # tensorboard
    parser.add_argument('--ex', default='test', help='experiment name')

    # 파라미터 저장
    opt = parser.parse_args()

    # tensorboard log폴더 생성
    opt.logdir = os.path.join('tensorboard', opt.ex)
    if not os.path.exists(opt.logdir): 
        os.makedirs(opt.logdir)

    return opt

if __name__ == '__main__':
    opt = get_params()
    

