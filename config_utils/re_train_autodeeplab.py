import argparse


def obtain_retrain_autodeeplab_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Autodeeplabv3+ Training")
    parser.add_argument('--train', action='store_true',
                        default=True, help='training mode')
    parser.add_argument('--exp', type=str, default='bnlr7e-3',
                        help='name of experiment')
    parser.add_argument('--gpu', type=str, default='0',
                        help='test time gpu device id')
    parser.add_argument('--backbone', type=str,
                        default='autodeeplab', help='resnet101')
    parser.add_argument('--dataset', type=str, default='solis',
                        help='solis, pascal or cityscapes')
    parser.add_argument('--groups', type=int, default=None,
                        help='num of groups for group normalization')
    parser.add_argument('--epochs', type=int, default=4000,
                        help='num of training epochs')
    parser.add_argument('--batch_size', type=int,
                        default=140, help='batch size')
    parser.add_argument('--base_lr', type=float,
                        default=0.05, help='base learning rate')
    parser.add_argument('--warmup_start_lr', type=float,
                        default=5e-6, help='warm up learning rate')
    parser.add_argument('--lr-step', type=float, default=None)
    parser.add_argument('--warmup-iters', type=int, default=1000)
    parser.add_argument('--min-lr', type=float, default=None)
    parser.add_argument('--last_mult', type=float, default=1.0,
                        help='learning rate multiplier for last layers')
    parser.add_argument('--scratch', action='store_true',
                        default=False, help='train from scratch')
    parser.add_argument('--freeze_bn', action='store_true',
                        default=False, help='freeze batch normalization parameters')
    parser.add_argument('--weight_std', action='store_true',
                        default=False, help='weight standardization')
    parser.add_argument('--beta', action='store_true',
                        default=False, help='resnet101 beta')
    parser.add_argument('--crop_size', type=int,
                        default=769, help='image crop size')
    parser.add_argument('--resize', type=int, default=769,
                        help='image crop size')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        metavar='S', help='random seed (default: 42)')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--filter_multiplier', type=int, default=32)
    parser.add_argument('--dist', type=bool, default=False)
    parser.add_argument('--autodeeplab', type=str, default='train')
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--use-ABN', default=True,
                        type=bool, help='whether use ABN')
    parser.add_argument('--affine', default=False, type=bool,
                        help='whether use affine in BN')
    parser.add_argument('--port', default=6000, type=int)
    parser.add_argument('--max-iteration', default=1000000, type=bool)
    parser.add_argument('--net_arch', default=None, type=str)
    parser.add_argument('--cell_arch', default=None, type=str)
    parser.add_argument('--criterion', default='ce', type=str)
    parser.add_argument('--initial-fm', default=None, type=int)
    parser.add_argument('--mode', default='poly',
                        type=str, help='how lr decline')
    parser.add_argument('--local_rank', dest='local_rank',
                        type=int, default=-1, )
    parser.add_argument('--train_mode', type=str,
                        default='iter', choices=['iter', 'epoch'])
    parser.add_argument('--num_bands', type=int, default=12)
    parser.add_argument('--checkname', type=str, default='')
    parser.add_argument('--num_images', type=int, default=None)
    parser.add_argument('--use_ab', type=bool, default=False)
    parser.add_argument('--subset_ratio', type=float,
                        default=None, help='ratio of images from the total solis dataset')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--use_new_model', type=bool, default=False)

    args = parser.parse_args()
    return args
