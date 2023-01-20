import torch
from utils import get_config, get_log_dir, get_cuda, str2bool
from data_loader import get_loader
from train import Trainer
import warnings
from tensorboardX import SummaryWriter
warnings.filterwarnings('ignore')
resume = ''

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Parameters to set
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        choices=['train', 'val'])
    parser.add_argument("--root_dataset",
                        type=str,
                        default='datasets/modelnet40_normal_resampled/')
    parser.add_argument("--resume", type=str, default='')
    parser.add_argument("--optimizer",
                        type=str,
                        default='Adam',
                        choices=['Adam', 'SGD'])
    parser.add_argument("--batch_size",
                        type=int,
                        default=2, help='batch size has to be greater than 1 since we use BN')
    parser.add_argument("--log_dir",
                        type=str,
                        default='logs/randomppf/')
    parser.add_argument("--cuda",
                        type=str,
                        default='True',
                        choices={'True','False'})
    parser.add_argument('--num_point', 
                        type=int, 
                        default=1024, 
                        help='Point Number')
    parser.add_argument('--use_normals', 
                        action='store_true', 
                        default=True, 
                        help='use normals')
    parser.add_argument('--use_uniform_sample', 
                        action='store_true', 
                        default=False, 
                        help='use uniform sampiling')
    parser.add_argument('--num_category', 
                        default=40, 
                        type=int, 
                        choices=[10, 40],  
                        help='Number of categories for training')
    opts = parser.parse_args()
    cfg = get_config()[1]
    opts.cfg = cfg

    if opts.mode in ['train']:
        opts.out = get_log_dir(opts.log_dir, cfg)
        print('Output logs: ', opts.out)

    data = get_loader(opts)

    # Setup visualization
    #vis = Visualizer(port=opts.log_port,
    #                 env='gearshaft')
    #if vis is not None:  # display options
    #    vis.vis_table("Options", vars(opts)
    vis = SummaryWriter(logdir=opts.log_dir+'tflogger/')
    b_size = int(opts.batch_size)
    trainer = Trainer(data, opts, b_size, vis)
    if opts.mode == 'val':
        trainer.Test()
    else:
        trainer.Train()
