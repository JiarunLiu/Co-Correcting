from os.path import join

import h5py
import numpy as np
from PIL import Image
import torch.utils.data
from dataset.utils import noisify


class PatchCamelyon(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 train=0,
                 balance=False,
                 redux=None,
                 random_ind_redux=False
                 ):
        self.root = root
        self.class_num = 2
        self.transform = transform
        self.target_transform = target_transform
        self.noise_type = noise_type
        self.random_state = 0

        ll = ['train', 'test', 'valid']
        self.meta_dir = join(root, "camelyonpatch_level_2_split_{}_meta.csv".format(ll[train]))
        self.x_dir = join(root, "camelyonpatch_level_2_split_{}_x.h5".format(ll[train]))
        self.y_dir = join(root, "camelyonpatch_level_2_split_{}_y.h5".format(ll[train]))

        with h5py.File(self.x_dir, "r") as f:
            self.data = np.array(f['x'])
        with h5py.File(self.y_dir, "r") as f:
            self.labels = np.array(f['y']).squeeze().astype(np.long)

        if balance:
            sample_num = []
            class_ind = []
            for i in range(self.class_num):
                class_ind_ = np.where(self.labels == i)[0]
                class_ind.append(class_ind_)
                sample_num.append(len(class_ind_))
            min_class_num = min(sample_num)

            new_labels = np.zeros(min_class_num * self.class_num, dtype=self.labels.dtype) - 1
            shape = [0,0,0,0]
            shape[1:] = self.data.shape[1:]
            shape[0] = min_class_num * self.class_num
            new_data = np.zeros(shape, dtype=self.data.dtype)

            for i in range(self.class_num):
                ind = np.random.choice(class_ind[i], min_class_num)
                new_labels[i * min_class_num:((i + 1) * min_class_num)] = self.labels[ind]
                new_data[i * min_class_num:((i + 1) * min_class_num)] = self.data[ind]
            self.labels = new_labels
            self.data = new_data

        if redux:
            if random_ind_redux:
                ind_new = np.random.choice(np.arange(self.labels.shape[0]), redux)
            else:
                ind_new = np.load(join(root, "{}_ind_redux.npy".format(ll[train])))
            self.labels = self.labels[ind_new]
            self.data = self.data[ind_new]
            print("Sample Number: 0 [{}] 1 [{}]".format(len(np.where(self.labels == 0)[0]), len(np.where(self.labels == 1)[0])))

        # noisy labels
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=np.bool)
        else:
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="PatchCamelyon",
                                                                nb_classes=self.class_num,
                                                                train_labels=np.expand_dims(self.labels, 1),
                                                                noise_type=noise_type,
                                                                noise_rate=noise_rate,
                                                                random_state=self.random_state)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target，index) where target is index of the target class.
        """
        img = self.data[index]
        target = self.labels[index] if self.noise_type == 'clean' else self.noisy_labels[index]

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.labels)