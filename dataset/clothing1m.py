import os
import os.path
from os.path import join

import numpy as np
from PIL import Image
import torch.utils.data
import torchvision.transforms as transforms

from dataset.utils import noisify


class Clothing1M(torch.utils.data.Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None,
                 noise_type='clean',
                 noise_rate=0.00,
                 device=0,
                 image_size=None,
                 balance=False
                 ):
        self.root = root
        self.class_num = 14
        self.transform = transform
        self.target_transform = target_transform
        self.device = device  # 0: hardware; 1: RAM
        self.noise_type = noise_type
        self.random_state = 0

        if image_size == None:
            self.imageTransform = transforms.Compose([
                transforms.RandomCrop((256, 256), pad_if_needed=True),
                transforms.Resize((224, 224), interpolation=Image.NEAREST)
            ])
        else:
            self.imageTransform = transforms.Compose([
                transforms.RandomCrop((256,256), pad_if_needed=True),
                transforms.Resize((image_size,image_size), interpolation=Image.NEAREST)
            ])

        img_folder_list = os.listdir(root)
        self.data = []
        self.labels = []
        for label_ in img_folder_list:
            imgs = os.listdir(join(root, label_))
            for img in imgs:
                if img[-3:] == 'jpg':
                    data_ = join(root, label_, img)
                    assert os.path.isfile(data_)
                    if self.device == 1:
                        data_ = self.img_loader(data_)
                    self.data.append(data_)
                    self.labels.append(int(label_))
        self.labels = np.asarray(self.labels)

        if self.device == 1:
            self.data = np.concatenate(self.data)
        else:
            self.data = np.array(self.data)

        if balance:
            sample_num = []
            class_ind = []
            for i in range(self.class_num):
                class_ind_ = np.where(self.labels == i)[0]
                class_ind.append(class_ind_)
                sample_num.append(len(class_ind_))
            min_class_num = min(sample_num)

            new_labels = np.zeros(min_class_num * self.class_num, dtype=self.labels.dtype) - 1
            if self.device == 1:
                shape = self.data.shape
                shape[0] = min_class_num * self.class_num
                new_data = np.zeros(shape, dtype=self.data.dtype)
            else:
                new_data = np.zeros((min_class_num * self.class_num), dtype=self.data.dtype)

            for i in range(self.class_num):
                ind = np.random.choice(class_ind[i], min_class_num)
                new_labels[i * min_class_num:((i + 1) * min_class_num)] = self.labels[ind]
                new_data[i * min_class_num:((i + 1) * min_class_num)] = self.data[ind]
            self.labels = new_labels
            self.data = new_data

        if self.device != 1:
            self.data = self.data.tolist()

        # noisy labels
        if noise_type == 'clean':
            self.noise_or_not = np.ones([len(self.labels)], dtype=np.bool)
        else:
            self.noisy_labels, self.actual_noise_rate = noisify(dataset="clothing1m",
                                                                nb_classes=14,
                                                                train_labels=np.expand_dims(self.labels, 1),
                                                                noise_type=noise_type,
                                                                noise_rate=noise_rate,
                                                                random_state=self.random_state)
            self.noisy_labels = self.noisy_labels.squeeze()
            self.noise_or_not = self.noisy_labels == self.labels

    def img_loader(self, img_path):
        return np.asarray(self.imageTransform(Image.open(img_path).convert("RGB"))).astype(np.uint8)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target，index) where target is index of the target class.
        """
        # img = self.img_loader(self.data[index]) if self.device == 0 else self.data[index]
        img = self.img_loader(self.data[index])
        target = self.labels[index] if self.noise_type == 'clean' else self.noisy_labels[index]
        # try:
        #     assert type(img) == np.ndarray
        #     assert img.shape == (224,224,3)
        # except:
        #     print("ERROR Data index: {}\tfile: {}\tshape: {}\ttype: {}".format(index, self.data[index], img.shape, type(img)))

        # doing this so that it is consistent with all other datasets to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    dataset = Clothing1M("/media/fgldlb/WD_4T/dataset/Clothing-1M/noisy_train",
                         transform=transforms.ToTensor(), balance=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=48,
                                             shuffle=True, num_workers=0)


    for i, (input, target, index) in enumerate(dataloader):
        print("i: {}\t"
              "input: {}\t"
              "target: {}\t"
              "index: {}".format(i,
                               input.shape,
                               target.shape,
                               index.shape))