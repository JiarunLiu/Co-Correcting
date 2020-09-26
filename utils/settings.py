import sys
import argparse

def get_args():
    if sys.platform == 'darwin':
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/ISIC-Archive-Downloader/Data_sample_balanced'
        mnist_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/MNIST'
        cifar10_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar10'
        cifar100_root = '/Users/jiarunliu/Documents/BUCT/Label_517/dataset/cifar/cifar100'
        pcam_root = "/Users/jiarunliu/Documents/BUCT/Label_517/dataset/PatchCamelyon"
        batch_size = 8
        device = 'cpu'
        data_device = 0
        noise_type = 'sn'
        stage1 = 1
        stage2 = 3
    elif sys.platform == 'linux':
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = '/home/fgldlb/Documents/ISIC-Archive-Downloader/NewData'
        pcam_root = "/home/fgldlb/Documents/dataset/PatchCamelyon"
        mnist_root = './data/mnist'
        cifar10_root = './data/cifar10'
        cifar100_root = './data/cifar100'
        batch_size = 32
        device = 'cuda:0'
        data_device = 1
        noise_type = 'sn'
        stage1 = 70
        stage2 = 200
    else:
        clothing1m_root = "/home/fgldlb/Documents/dataset/Clothing-1M"
        isic_root = None
        mnist_root = './data/mnist'
        cifar10_root = '/data/cifar10'
        cifar100_root = '/data/cifar100'
        pcam_root = None
        batch_size = 16
        device = 'cpu'
        data_device = 0
        noise_type = 'clean'
        stage1 = 70
        stage2 = 200

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # normal parameters
    parser.add_argument('-b', '--batch-size', default=batch_size, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='H-P', help='initial learning rate')
    parser.add_argument('--lr2', '--learning-rate2', default=1e-5, type=float,
                        metavar='H-P', help='initial learning rate of stage3')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--backbone', dest="backbone", default="resnet50", type=str,
                        help="backbone for PENCIL training")
    parser.add_argument('--optim', dest="optim", default="SGD", type=str,
                        choices=['SGD', 'Adam', 'AdamW', 'RMSprop', 'Adadelta', 'Adagrad', 'mix'],
                        help="Optimizer for PENCIL training")
    parser.add_argument('--scheduler', dest='scheduler', default=None, type=str, choices=['cyclic', None, "SWA"],
                        help="Optimizer for PENCIL training")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    # Co-teaching parameters 1
    parser.add_argument('--forget-rate', '--fr', '--forget_rate', default=0.2, type=float,
                        metavar='H-P', help='Forget rate. Suggest same with noisy density.')
    parser.add_argument('--num-gradual', '--ng', '--num_gradual', default=10, type=int,
                        metavar='H-P', help='how many epochs for linear drop rate, can be 5, 10, 15. '
                                            'This parameter is equal to Tk for R(T) in Co-teaching paper.')
    parser.add_argument('--exponent', default=1, type=float,
                        metavar='H-P', help='exponent of the forget rate, can be 0.5, 1, 2. '
                                            'This parameter is equal to c in Tc for R(T) in Co-teaching paper.')
    parser.add_argument('--loss-type', dest="loss_type", default="coteaching_plus", type=str,
                        choices=['coteaching_plus', 'coteaching'],
                        help="loss type: [coteaching_plus, coteaching]")
    parser.add_argument('--warmup', '--wm', '--warm-up', default=0, type=float,
                        metavar='H-P', help='Warm up process eopch, default 0.')
    parser.add_argument('--linear-num', '--linear_num', default=256, type=int,
                        metavar='H-P', help='how many epochs for linear drop rate, can be 5, 10, 15. '
                                            'This parameter is equal to Tk for R(T) in Co-teaching paper.')
    # PENCIL parameters 1
    parser.add_argument('--alpha', default=0.4, type=float,
                        metavar='H-P', help='the coefficient of Compatibility Loss')
    parser.add_argument('--beta', default=0.1, type=float,
                        metavar='H-P', help='the coefficient of Entropy Loss')
    parser.add_argument('--lambda1', default=200, type=int,
                        metavar='H-P', help='the value of lambda, ')
    parser.add_argument('--K', default=10.0, type=float, )
    # PENCIL parameters 2
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=320, type=int, metavar='H-P',
                        help='number of total epochs to run')
    parser.add_argument('--stage1', default=stage1, type=int,
                        metavar='H-P', help='number of epochs utill stage1')
    parser.add_argument('--stage2', default=stage2, type=int,
                        metavar='H-P', help='number of epochs utill stage2')
    # Nosie settings
    parser.add_argument('--noise', default=0.20, type=float,
                        help='noise density of data label')
    parser.add_argument('--noise_type', default=noise_type,  choices=['clean', 'sn', 'pairflip'],type=str,
                        help='noise tyoe of data label')
    # Data settings 1
    parser.add_argument("--dataset", dest="dataset", default='mnist', type=str,
                        choices=['mnist', 'cifar10', 'cifar100', 'cifar2', 'isic', 'clothing1m', 'pcam'],
                        help="model input image size")
    parser.add_argument("--image_size", dest="image_size", default=224, type=int,
                        help="model input image size")
    parser.add_argument('--classnum', default=2, type=int,
                        metavar='H-P', help='number of train dataset classes')
    parser.add_argument('--device', dest='device', default=device, type=str,
                        help='select gpu')
    parser.add_argument('--data_device', dest="data_device", default=data_device, type=int,
                        help="Dataset loading device, 0 for hardware 1 for RAM. Default choice is 1. "
                             "Please ensure your computer have enough capacity!")
    parser.add_argument('--dataRoot',dest='root',default=isic_root,
                        type=str,metavar='PATH',help='where is the dataset')
    # Data settings 2
    parser.add_argument('--datanum', default=15000, type=int,
                        metavar='H-P', help='number of train dataset samples')
    parser.add_argument('--train-redux', dest="train_redux", default=5120, type=int,
                        help='train data number, default None')
    parser.add_argument('--test-redux', dest="test_redux", default=1280, type=int,
                        help='test data number, default None')
    parser.add_argument('--val-redux', dest="val_redux", default=1280, type=int,
                        help='validate data number, default None')
    parser.add_argument('--full-test', dest="full_test", default=False, type=bool,
                        help='use full test set data, default False')
    parser.add_argument('--random-ind-redux', dest="random_ind_redux", default=False, type=bool,
                        help='use full test set data, default False')
    # Curriculum settings
    parser.add_argument("--curriculum", dest="curriculum", default=1, type=int,
                        help="curriculum in label updating")
    parser.add_argument("--cluster-mode", dest="cluster_mode", default='dual', type=str, choices=['dual', 'single', 'dual_PCA'],
                        help="curriculum in label updating")
    parser.add_argument("--shuffle-label", dest="shuffle_label", default=0, type=float,
                        help="shuffle-label in label updating")
    parser.add_argument("--dim-reduce", dest="dim_reduce", default=256, type=int,
                        help="Curriculum features dim reduce by PCA")
    parser.add_argument("--mix-grad", dest="mix_grad", default=1, type=int,
                        help="mix gradient of two-stream arch, 1=True")
    parser.add_argument("--discard", dest="discard", default=1, type=int,
                        help="only update discard sample's label, 1=True")
    parser.add_argument("--gamma", dest="gamma", default=0.6, type=int,
                        help="forget rate schelduler param")
    parser.add_argument("--finetune-schedule", '-fs', dest="finetune_schedule", default=0, type=int,
                        help="forget rate schelduler param")
    # trainer settings
    parser.add_argument('--dir', dest='dir', default="experiment/test-debug", type=str,
                        metavar='PATH', help='save dir')
    parser.add_argument('--random-seed', dest='random_seed', default=None, type=int,
                        metavar='N', help='pytorch random seed, default None.')
    # parser.add_argument('--tips', "--tip", dest="tip", default='', type=str,
    #                     help="Training tips, just record in json file.")


    args = parser.parse_args()

    # Setting for different dataset
    if args.dataset == "isic":
        print("Training on ISIC")
        args.backbone = 'resnet50'
        args.image_size = 224
        args.classnum = 2
        args.input_dim = 3
    elif args.dataset == 'mnist':
        print("Training on mnist")
        args.backbone = 'cnn'
        if args.root == isic_root:
            args.root = mnist_root
        args.batch_size = 128
        args.image_size = 28
        args.classnum = 10
        args.input_dim = 1
        args.linear_num = 144
        args.datanum = 60000
        args.lr = 0.001
        args.lr2 = 0.0001
    elif args.dataset == 'cifar10':
        print("Training on cifar10")
        # args.backbone = 'cnn'
        if args.root == isic_root:
            args.root = cifar10_root
        args.warmup = 0
        args.batch_size = 128
        args.image_size = 32
        args.classnum = 10
        args.input_dim = 3
        args.datanum = 50000
    elif args.dataset == 'cifar100':
        print("Training on cifar100")
        args.backbone = 'cnn'
        if args.root == isic_root:
            args.root = cifar100_root
        args.batch_size = 128
        args.image_size = 32
        args.classnum = 100
        args.input_dim = 3
        args.datanum = 50000
    elif args.dataset == 'clothing1m':
        if args.root == isic_root:
            args.root = clothing1m_root
        args.data_device = 0
        args.backbone = 'resnet50'
        args.image_size = 224
        args.classnum = 14
        args.input_dim = 3
        args.datanum = 1000000
        args.stage1 = 5
        args.stage2 = 15
        args.epochs = 25
        args.batch_size = 32
        args.dim_reduce = 256
        args.noise_type = 'clean'
    elif args.dataset == 'pcam':
        if args.root == isic_root:
            args.root = pcam_root
        args.backbone = 'densenet169'
        args.batch_size = 128
        args.image_size = 96
        args.dim_reduce = 128
        args.classnum = 2
        args.input_dim = 3
        args.stage1 = 70
        args.stage2 = 200
        args.epochs = 320
        args.datanum = 262144
        args.train_redux = 26214
        args.test_redux = 3276
        args.val_redux = 3276
        args.random_ind_redux = False
    else:
        print("Use default setting")

    return args