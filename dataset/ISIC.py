
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import os.path
from os.path import join
import json

from dataset.utils import noisify

class ISIC(torch.utils.data.Dataset):
    '''
    Notices:
    train/val/test : first 60%/10%/30%
    hxw:767 x 1022
    '''

    def __init__(self,
                 root,
                 train=0,
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 device=1,
                 redux=None,
                 image_size=None
                 ):
        base_folder = root
        self.image_folder = join(base_folder, 'Images')
        self.data_list_f = join(base_folder, "data_list2.json")
        self.label_folder = join(base_folder, 'Descriptions')

        self.labelOrder = ['benign', 'malignant']
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.device = device  # 0: hardware; 1: RAM
        self.noise_type = noise_type
        self.random_state = 0

        with open(self.data_list_f, 'r') as data_f:
            data_dict = json.load(data_f)

        if self.train == 0:
            self.data_list = data_dict['train']
        elif self.train == 1:
            self.data_list = data_dict['test']
        else:
            self.data_list = data_dict['val']

        if redux:
            self.data_list = self.data_list[:redux]

        if image_size == None:
            self.imageTransform = transforms.Compose([
                transforms.Resize((720, 720), interpolation=Image.NEAREST)
            ])
        else:
            self.imageTransform = transforms.Compose([
                transforms.Resize((image_size,image_size), interpolation=Image.NEAREST)
            ])

        print("Loading data from {}".format(self.label_folder))
        # now load the picked numpy arrays
        self.data = []
        self.labels = []
        for f in self.data_list:
            file = join(self.label_folder, f)
            ff = open(file)
            entry = json.load(ff)
            try:
                flabel = entry['meta']['clinical']['benign_malignant']
                if not flabel in self.labelOrder:
                    raise Exception
                label_ = self.labelOrder.index(flabel)
            except:
                label_ = 0  # All 19 kinds,0-17 normal label, 18 as exception
            data_ = join(self.image_folder, f + '.jpeg')
            #print(data_)
            assert os.path.isfile(data_)
            if self.device == 1:
                data_ = self.img_loader(data_)
            self.data.append(data_)
            self.labels.append(label_)

        if self.device == 1:
            self.data == np.concatenate(self.data)

        # noisy labels
        self.labels = np.asarray(self.labels)
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=np.bool)
        else:
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="ISIC",
                                                                      nb_classes=2,
                                                                      train_labels=np.expand_dims(self.labels, 1),
                                                                      noise_type=noise_type,
                                                                      noise_rate=noise_rate,
                                                                      random_state=self.random_state)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def img_loader(self, img_path):
        return np.asarray(self.imageTransform(Image.open(img_path))).astype(np.uint8)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target，index) where target is index of the target class.
        """
        img = self.img_loader(self.data[index]) if self.device == 0 else self.data[index]
        target = self.labels[index] if self.noise_type == 'clean' else self.noisy_labels[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data_list)
