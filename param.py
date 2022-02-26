from argparse import *
import os

def get_params():
    # 파라미터를 받는 인스턴스 생성
    parser = ArgumentParser()

    # 데이터셋 파라미터
        # source
    parser.add_argument('--source', default='Cityscapes', help='source dataset')
    parser.add_argument('--source_list_path', default='./datasets/Cityscapes/list', help='source data .txt file path')
    parser.add_argument('--split_source_train', default='train', help='source train data | all | train | test | val')

        # target
    parser.add_argument('--target', default='Cityscapes', help='target dataset')
    parser.add_argument('--target_list_path', default='./datasets/Cityscapes/list', help='target data .txt file path')
    parser.add_argument('--split_target_train', default='train', help='target train data | all | train | test | val') 
    parser.add_argument('--split_target_test', default='val', help='target test data | all | train | test | val') 

        # common
    parser.add_argument('--img_size', default=(512,1024), help='(h,w), image size for train and test')

    # 학습 파라미터
    parser.add_argument('--batch_size', default=2, type=int, help='batch size')
    parser.add_argument('--shuffle', default=False, help='shuffle dataloader')
    parser.add_argument('--drop_last', default=True, help='drop last batch')
    parser.add_argument('--total_iter', default=100000, help='total iteration')
    parser.add_argument('--lr', default=2.5e-4, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=5.0e-4, type=float, help='weight_decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='weight_decay')
    parser.add_argument('--lr_schedule_freq', default=10000, type=int, help='frequency to update learning rate')
    parser.add_argument('--init_imgnet', default=True, type=bool, help='initialization from imgnet pretrained model')

    # 환경설정 파라미터
    parser.add_argument('--random_seed', default=3621, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, help='use cuda')

    # log & tensorboard
    parser.add_argument('--ex', default='test', help='experiment name')
    parser.add_argument('--tensor_freq', default=10, type=int, help='frequency to show result on tensorboard')

    # evaluation
    parser.add_argument('--eval_freq', default=100, type=int, help='frequency to evaluate model')

    # 파라미터 저장
    opt = parser.parse_args()

    # log 폴더 생성
    opt.log_dir = os.path.join('log', opt.ex)
    if not os.path.exists(opt.log_dir): 
        os.makedirs(opt.log_dir)

    return opt

if __name__ == '__main__':
    opt = get_params()
    

