import json
import numpy as np
from os.path import join

class Label_Checker(object):

    def __init__(self, clean_dir, list):
        self.clean_dir = clean_dir
        self.data_list = list
        self.load_gt()

    def load_gt(self):
        print("=> loading ground truth label from {}".format(self.clean_dir))

        self.labelOrder = ['benign', 'malignant']
        self.gt = []
        for f in self.data_list:
            file = join(self.clean_dir, f)
            with open(file, 'r') as ff:
                entry = json.load(ff)
            try:
                flabel = entry['meta']['clinical']['benign_malignant']
                if not flabel in self.labelOrder:
                    raise Exception
                label_ = self.labelOrder.index(flabel)
            except:
                label_ = 0
            self.gt.append(label_)
        self.gt = np.array(self.gt)
        print("=> ground truth label loaded")

    def check(self, predict):
        """
        predict: numpy() format, shape: (data_num, class)
        """
        # get max prob index
        predict = np.argmax(predict, axis=1)

        if len(predict) != len(self.gt):
            predict = predict[:len(self.gt)]

        t = (self.gt == predict).sum()
        accu = t / len(self.gt)
        return accu


def check_label_acc(A, B, onehotA=False, onehotB=False):
    """
    get correct label percent in all labels
    :param A: label A
    :param B: label B
    :param onehotA: bool, is label A in onehot?
    :param onehotB: bool, is label B in onehot?
    :return: matched percent in total labels
    """
    A = np.argmax(A, axis=1) if onehotA else A
    B = np.argmax(B, axis=1) if onehotB else B

    try:
        assert A.shape == B.shape
    except:
        redux = min(A.shape[0], B.shape[0])
        A = A[:redux]
        B = B[:redux]

    t = np.sum(A == B)
    accu = t / len(A)
    return accu

def check_label_noisy2true(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    new_label = np.argmax(new_label, axis=1) if onehotA else new_label
    clean_label = np.argmax(clean_label, axis=1) if onehotB else clean_label

    try:
        assert new_label.shape == clean_label.shape
    except:
        redux = min(new_label.shape[0], clean_label.shape[0])
        new_label = new_label[:redux]
        clean_label = clean_label[:redux]

    assert new_label.shape == noise_or_not.shape
    assert new_label.shape == clean_label.shape

    n2t_num = np.sum((new_label == clean_label).astype(np.int32) * (~noise_or_not).astype(np.int32))
    n2t = n2t_num / clean_label.shape[0]

    return n2t


def check_label_true2noise(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    new_label = np.argmax(new_label, axis=1) if onehotA else new_label
    clean_label = np.argmax(clean_label, axis=1) if onehotB else clean_label

    try:
        assert new_label.shape == clean_label.shape
    except:
        redux = min(new_label.shape[0], clean_label.shape[0])
        new_label = new_label[:redux]
        clean_label = clean_label[:redux]

    assert new_label.shape == noise_or_not.shape
    assert new_label.shape == clean_label.shape

    t2n_num = np.sum((new_label != clean_label).astype(np.int32) * noise_or_not.astype(np.int32))
    t2n = t2n_num / clean_label.shape[0]

    return t2n

def check_label(new_label, clean_label, noise_or_not, onehotA=False, onehotB=False):
    acc = check_label_acc(new_label, clean_label, onehotA, onehotB)
    n2t = check_label_noisy2true(new_label, clean_label, noise_or_not, onehotA, onehotB)
    t2n = check_label_true2noise(new_label, clean_label, noise_or_not, onehotA, onehotB)
    return acc, n2t, t2n

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data_list_f', type=str, metavar='N',
                        default="/home/fgldlb/Documents/ISIC-Archive-Downloader/Data_balanced/Descriptions/data_list.json",
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--clean_dir', type=str, metavar='N',
                        default="/home/fgldlb/Documents/ISIC-Archive-Downloader/Data_balanced/Descriptions/clean",
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--npy', type=str, metavar='N',
                        default="./../experiment/balance_resnet_PENCIL_sn_010/y.npy",
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()

    with open(args.data_list_f, 'r') as data_f:
        data_dict = json.load(data_f)
    train_list = data_dict['train']
    soft_label = np.load(args.npy)

    label_checker = Label_Checker(args.clean_dir, train_list)
    accu = label_checker.check(soft_label)
    print(accu)