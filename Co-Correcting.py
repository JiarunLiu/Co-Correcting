import os
import copy
import json
import time
import shutil
import numpy as np
from os.path import join

import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.decomposition import PCA

from utils.settings import get_args
from utils.label_checker import check_label
from utils.curriculum_clustering import CurriculumClustering

from Loss import Loss
from BasicTrainer import BasicTrainer


class CoCorrecting(BasicTrainer, Loss):
    """
    train co-pencil method
    """

    def __init__(self, args):
        super().__init__(args)

        # Initialize Cooperation Models
        self.modelA = self._get_model(self.args.backbone)
        self.modelB = self._get_model(self.args.backbone)
        self.modelC = self._get_model(self.args.backbone)
        # Optimizer & Criterion
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.logsoftmax = nn.LogSoftmax(dim=1).to(self.args.device)
        self.softmax = nn.Softmax(dim=1).to(self.args.device)
        self.optimizerA = self._get_optim(self.modelA.parameters(), optim=self.args.optim)
        self.optimizerB = self._get_optim(self.modelB.parameters(), optim=self.args.optim)
        # trainer init
        self._recoder_init()
        self._save_meta()
        # Load Data
        self.trainloader, self.testloader, self.valloader = self._load_data()
        # Optionally resume from a checkpoint
        if os.path.isfile(self.args.checkpoint_dir):
            self._resume()
        else:
            print("=> no checkpoint found at '{}'".format(self.args.checkpoint_dir))
            # save clean label & noisy label
            np.save(join(self.args.dir, 'y_clean.npy'), self.clean_labels)

    def _resume(self):
        # load model state
        print("=> loading checkpoint '{}'".format(self.args.checkpoint_dir))
        checkpoint = torch.load(self.args.checkpoint_dir)
        self.args.start_epoch = checkpoint['epoch']
        self.best_prec1 = checkpoint['best_prec1']
        self.modelA.load_state_dict(checkpoint['state_dict_A'])
        self.optimizerA.load_state_dict(checkpoint['optimizer_A'])
        self.modelB.load_state_dict(checkpoint['state_dict_B'])
        self.optimizerB.load_state_dict(checkpoint['optimizer_B'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(self.args.checkpoint_dir, checkpoint['epoch']))

        # load record_dict
        if os.path.isfile(self.args.record_dir):
            print("=> loading record file {}".format(self.args.record_dir))
            with open(self.args.record_dir, 'r') as f:
                self.record_dict = json.load(f)
                print("=> loaded record file {}".format(self.args.record_dir))

    def _recoder_init(self):
        os.makedirs(self.args.dir, exist_ok=True)
        os.makedirs(join(self.args.dir, 'record'), exist_ok=True)

        keys = ['acc', 'acc5', 'label_accu', 'loss', "pure_ratio", "label_n2t", "label_t2n", "pure_ratio_discard"]
        record_infos = {}
        for k in keys:
            record_infos[k] = []
        # 3 is mixed model result
        self.record_dict = {
            'train1': copy.deepcopy(record_infos),
            'test1': copy.deepcopy(record_infos),
            'val1': copy.deepcopy(record_infos),
            'train2': copy.deepcopy(record_infos),
            'test2': copy.deepcopy(record_infos),
            'val2': copy.deepcopy(record_infos),
            'train3': copy.deepcopy(record_infos),
            'test3': copy.deepcopy(record_infos),
            'val3': copy.deepcopy(record_infos),
            'loss_val': [],
            'loss_avg': [],
        }

    def _record(self):
        # write file
        with open(self.args.record_dir, 'w') as f:
            json.dump(self.record_dict, f, indent=4, sort_keys=True)

    # define drop rate schedule
    def gen_forget_rate(self, forget_rate, fr_type='type_1'):
        if fr_type == 'type_1':
            rate_schedule = np.ones(args.n_epoch) * forget_rate
            rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)

        if fr_type == 'type_2':
            rate_schedule = np.ones(args.n_epoch) * forget_rate
            rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate, args.num_gradual)
            rate_schedule[args.num_gradual:] = np.linspace(forget_rate, 2 * forget_rate,
                                                           args.n_epoch - args.num_gradual)

        return rate_schedule

    def _rate_schedule(self, epoch):
        rate_schedule = np.ones(self.args.epochs) * self.args.forget_rate
        if self.args.warmup > 0:
            rate_schedule[:self.args.warmup] = 0
            rate_schedule[self.args.warmup:self.args.warmup+self.args.num_gradual] = np.linspace(self.args.warmup,
                                                                self.args.warmup + (
                                                                            self.args.forget_rate ** self.args.exponent),
                                                                self.args.num_gradual)
        else:
            rate_schedule[:self.args.num_gradual] = np.linspace(0,
                                                                self.args.forget_rate ** self.args.exponent,
                                                                self.args.num_gradual)
        if self.args.finetune_schedule == 1:
            rate_schedule[self.args.stage2:] = np.linspace(self.args.forget_rate ** self.args.exponent,
                                                           self.args.forget_rate ** self.args.gamma,
                                                           self.args.epochs - self.args.stage2)
        return rate_schedule[epoch]

    def _adjust_learning_rate(self, epoch):
        """Sets the learning rate"""
        if epoch < self.args.stage2:
            lr = self.args.lr
        elif epoch < (self.args.epochs - self.args.stage2) // 3 + self.args.stage2:
            lr = self.args.lr2
        elif epoch < 2 * (self.args.epochs - self.args.stage2) // 3 + self.args.stage2:
            lr = self.args.lr2 / 10
        else:
            lr = self.args.lr2 / 100
        for param_group in self.optimizerA.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizerB.param_groups:
            param_group['lr'] = lr

    def _compute_loss(self, outputA, outputB, target, target_var, index, epoch, i, parallel=False):
        if epoch < self.args.stage1:
            # init y_tilde, let softmax(y_tilde) is noisy labels
            onehot = torch.zeros(target.size(0),
                                 self.args.classnum).scatter_(1, target.view(-1, 1), self.args.K)
            onehot = onehot.numpy()
            self.new_y[index, :] = onehot
            # training as normal co-teaching
            forget_rate = self._rate_schedule(epoch)
            if self.args.loss_type == 'coteaching':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching(outputA, outputB, target_var, target_var,
                      forget_rate, ind=index, loss_type='CE', noise_or_not=self.noise_or_not, softmax=True)
            elif self.args.loss_type == 'coteaching_plus':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching_plus(outputA, outputB, target_var, target_var,
                    forget_rate, epoch * i, index, loss_type='CE', noise_or_not=self.noise_or_not, softmax=False)
            else:
                raise NotImplementedError("loss_type {} not been found".format(self.args.loss_type))
            return lossA, lossB, onehot, onehot, ind_A_discard, ind_B_discard, \
                   pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2
        elif epoch < self.args.stage2:
            # using select data sample update parameters, other update label only
            yy_A = self.yy
            yy_B = self.yy
            yy_A = torch.tensor(yy_A[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            yy_B = torch.tensor(yy_B[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            # obtain label distributions (y_hat)
            last_y_var_A = self.softmax(yy_A)
            last_y_var_B = self.softmax(yy_B)
            # sort samples
            forget_rate = self._rate_schedule(epoch)
            if self.args.loss_type == 'coteaching':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, \
                pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching(
                    outputA, outputB, last_y_var_A, last_y_var_B, forget_rate, ind=index, loss_type="PENCIL",
                    target_var=target_var, noise_or_not=self.noise_or_not, parallel=parallel, softmax=False)
            elif self.args.loss_type == 'coteaching_plus':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, \
                pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching_plus(
                    outputA, outputB, last_y_var_A, last_y_var_B, forget_rate, epoch * i, index, loss_type="PENCIL",
                    target_var=target_var, noise_or_not=self.noise_or_not, parallel=parallel, softmax=False)
            else:
                raise NotImplementedError("loss_type {} not been found".format(self.args.loss_type))
            return lossA, lossB, yy_A, yy_B, ind_A_discard, ind_B_discard, \
                   pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2
        else:
            yy_A = self.yy
            yy_A = torch.tensor(yy_A[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            yy_B = self.yy
            yy_B = torch.tensor(yy_B[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
            last_y_var_A = self.softmax(yy_A)
            last_y_var_B = self.softmax(yy_B)
            forget_rate = self._rate_schedule(epoch)
            if self.args.loss_type == 'coteaching':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching_plus(outputA, outputB,
                                                                                       last_y_var_A, last_y_var_B,
                                                                                       forget_rate,
                                                                                       ind=index,
                                                                                       loss_type="PENCIL_KL",
                                                                                       noise_or_not=self.noise_or_not,
                                                                                       softmax=False)
            elif self.args.loss_type == 'coteaching_plus':
                lossA, lossB, ind_A_update, ind_B_update, ind_A_discard, ind_B_discard, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self.loss_coteaching_plus(outputA, outputB,
                                                                                       last_y_var_A, last_y_var_B,
                                                                                       forget_rate, epoch * i,
                                                                                       index,
                                                                                       loss_type="PENCIL_KL",
                                                                                       noise_or_not=self.noise_or_not,
                                                                                       softmax=False)
            else:
                raise NotImplementedError("loss_type {} not been found".format(self.args.loss_type))

            return lossA, lossB, yy_A, yy_B, ind_A_discard, ind_B_discard, \
                   pure_ratio_1, pure_ratio_2, pure_ratio_discard_1, pure_ratio_discard_2

    def _gen_clustering_data(self, mode='dual'):
        if mode == 'dual':
            featureA = []
            def hookA(module, input, output):
                featureA.append(output.clone().cpu().detach())

            featureB = []
            def hookB(module, input, output):
                featureB.append(output.clone().cpu().detach())

            self.modelA.eval()
            with torch.no_grad():
                layer_num = 0
                for i in self.modelA.modules():
                    layer_num += 1
                target_layer_ind = layer_num - 2
                for i, j in enumerate(self.modelA.modules()):
                    if i == target_layer_ind:
                        handleA = j.register_forward_hook(hookA)
                for i, j in enumerate(self.modelB.modules()):
                    if i == target_layer_ind:
                        handleB = j.register_forward_hook(hookB)

                # guess feature num
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    break
                assert featureA[0].shape == featureB[0].shape
                num_features = featureA[0].view(featureA[0].size(0), -1).shape[-1]
                featureA = []
                featureB = []

                features = np.zeros((self.train_data_num, num_features * 2), dtype=np.float32)
                labels = np.zeros(self.train_data_num, dtype=np.long)

                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    features[index] = np.concatenate((featureA[i].view(featureA[i].size(0), -1).numpy(),
                                        featureB[i].view(featureB[i].size(0), -1).numpy()),
                                       axis=1)
                    labels[index] = target
                    print("\rget clustering data: [{}/{}]".format(i+1, self.train_batch_num), end='')
                handleA.remove()
                handleB.remove()
                print("\nFinish collect cluster data.")

        elif mode == 'single':
            featureA = []

            def hookA(module, input, output):
                featureA.append(output.clone().cpu().detach())

            self.modelA.eval()
            with torch.no_grad():
                layer_num = 0
                for i in self.modelA.modules():
                    layer_num += 1
                target_layer_ind = layer_num - 2
                for i, j in enumerate(self.modelA.modules()):
                    if i == target_layer_ind:
                        handleA = j.register_forward_hook(hookA)

                # guess feature num
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    break
                num_features = featureA[0].view(featureA[0].size(0), -1).shape[-1]
                featureA = []

                features = np.zeros((self.train_data_num, num_features), dtype=np.float32)
                labels = np.zeros(self.train_data_num, dtype=np.long)

                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    features[index] = featureA[i].view(featureA[i].size(0), -1).numpy()
                    labels[index] = target
                    print("\rget clustering data: [{}/{}]".format(i + 1, self.train_batch_num), end='')
                handleA.remove()
                print("\nFinish collect cluster data.")

        elif mode == 'dual_PCA':
            featureA = []
            def hookA(module, input, output):
                featureA.append(output.clone().cpu().detach())

            featureB = []
            def hookB(module, input, output):
                featureB.append(output.clone().cpu().detach())

            self.modelA.eval()
            with torch.no_grad():
                layer_num = 0
                for i in self.modelA.modules():
                    layer_num += 1
                target_layer_ind = layer_num - 2
                for i, j in enumerate(self.modelA.modules()):
                    if i == target_layer_ind:
                        handleA = j.register_forward_hook(hookA)
                for i, j in enumerate(self.modelB.modules()):
                    if i == target_layer_ind:
                        handleB = j.register_forward_hook(hookB)

                # guess feature num
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    break
                assert featureA[0].shape == featureB[0].shape
                num_features = featureA[0].view(featureA[0].size(0), -1).shape[-1]
                featureA = []
                featureB = []

                features = np.zeros((self.train_data_num, self.args.dim_reduce), dtype=np.float32)
                labels = np.zeros(self.train_data_num, dtype=np.long)

                featureA_ = np.zeros((self.train_data_num, num_features), dtype=np.float32)
                featureB_ = np.zeros((self.train_data_num, num_features), dtype=np.float32)
                for i, (input, target, index) in enumerate(self.trainloader):
                    input = input.to(self.args.device)
                    _ = self.modelA(input)
                    _ = self.modelB(input)
                    featureA_[index] = featureA[i].view(featureA[i].size(0), -1).numpy()
                    featureB_[index] = featureB[i].view(featureB[i].size(0), -1).numpy()
                    labels[index] = target
                    print("\rget clustering data: [{}/{}]".format(i+1, self.train_batch_num), end='')

                target_features = self.args.dim_reduce//2

                pca = PCA(n_components=target_features, copy=False)
                featureA = pca.fit_transform(np.array(featureA_))
                featureB = pca.fit_transform(np.array(featureB_))

                features[:, target_features:] = featureA
                features[:, :target_features] = featureB

                handleA.remove()
                handleB.remove()
                print("\nFinish collect cluster data.")
        else:
            NotImplementedError("mode {} not been implemneted!!!".format(mode))

        return features, labels

    def _cluster_data_into_subsets(self):
        features, labels = self._gen_clustering_data(self.args.cluster_mode)
        cc = CurriculumClustering(n_subsets=3, verbose=True, random_state=0, dim_reduce=self.args.dim_reduce)
        cc.fit(features, labels)
        self.subset_labels = cc.output_labels
        np.save(join(self.args.dir, 'subset_labels.npy'), self.subset_labels)

    def _get_label_update_stage(self, epoch):
        update_stage = 0
        step = (self.args.stage2 - self.args.stage1) / 3
        for i in range(3):
            if self.args.stage1 + step * (i + 1) < epoch:
                update_stage += 1
        return update_stage

    def training(self):
        timer = AverageMeter()
        # train
        end = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            print('-----------------')

            self._adjust_learning_rate(epoch)

            # load y_tilde
            if os.path.isfile(self.args.y_file):
                self.yy = np.load(self.args.y_file)
            else:
                self.yy = []

            if epoch == self.args.stage1:
                self._cluster_data_into_subsets()

            if self.args.classnum > 5:
                train_prec1_A, train_prec1_B, train_prec5_A, train_prec5_B = self.train(epoch)
                val_prec1_A, val_prec1_B, val_prec5_A, val_prec5_B = self.val(epoch)
                test_prec1_A, test_prec1_B, test_prec5_A, test_prec5_B = self.test(epoch)
            else:
                train_prec1_A, train_prec1_B = self.train(epoch)
                val_prec1_A, val_prec1_B = self.val(epoch)
                test_prec1_A, test_prec1_B = self.test(epoch)

            # load best model to modelC
            if epoch < self.args.stage1:
                best_ind = [val_prec1_A, val_prec1_B, self.best_prec1].index(max(val_prec1_A, val_prec1_B, self.best_prec1))
                self.modelC.load_state_dict([self.modelA.state_dict(), self.modelB.state_dict(), self.modelC.state_dict()][best_ind])

            is_best = max(val_prec1_A, val_prec1_B) > self.best_prec1
            self.best_prec1 = max(val_prec1_A, val_prec1_B, self.best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.args.backbone,
                'state_dict_A': self.modelA.state_dict(),
                'optimizer_A': self.optimizerA.state_dict(),
                'state_dict_B': self.modelB.state_dict(),
                'optimizer_B': self.optimizerB.state_dict(),
                'prec_A': val_prec1_A,
                'prec_B': val_prec1_B,
                'best_prec1': self.best_prec1,
            }, is_best, filename=self.args.checkpoint_dir, modelbest=self.args.modelbest_dir)

            self._record()

            timer.update(time.time() - end)
            end = time.time()
            print("Epoch {} using {} min {:.2f} sec".format(epoch, timer.val // 60, timer.val % 60))

    def train(self, epoch=0):
        batch_time = AverageMeter()
        losses_A = AverageMeter()
        losses_B = AverageMeter()
        top1_A = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()
        pure_ratio_A = AverageMeter()
        pure_ratio_B = AverageMeter()
        pure_ratio_discard_A = AverageMeter()
        pure_ratio_discard_B = AverageMeter()
        margin_accu = AverageMeter()
        if self.args.classnum > 5:
            top5_A = AverageMeter()
            top5_B = AverageMeter()
            top5_mix = AverageMeter()
        end = time.time()

        self.modelA.train()
        self.modelB.train()

        # new y is y_tilde after updating
        self.new_y = np.zeros([self.args.datanum, self.args.classnum])
        if epoch >= self.args.stage1:
            self.new_y = self.yy

        for i, (input, target, index) in enumerate(self.trainloader):

            index = index.numpy()

            input = input.to(self.args.device)
            target1 = target.to(self.args.device)
            input_var = input.clone().to(self.args.device)
            target_var = target1.clone().to(self.args.device)

            outputA = self.modelA(input_var)
            outputB = self.modelB(input_var)
            output_mix = (outputA + outputB) / 2

            if epoch < self.args.warmup:
                lossA = self._get_loss(outputA, target1, loss_type="CE")
                lossB = self._get_loss(outputB, target1, loss_type="CE")
                pure_ratio_1, pure_ratio_2 = 0, 0
                pure_ratio_discard_1, pure_ratio_discard_2 = 0, 0
            else:
                lossA, lossB, yy_A, yy_B, ind_A_discard, ind_B_discard, pure_ratio_1, pure_ratio_2, \
                pure_ratio_discard_1, pure_ratio_discard_2 = self._compute_loss(outputA, outputB, target, target_var,
                                                                                index, epoch, i)

            outputA_ = outputA
            outputB_ = outputB

            outputA = F.softmax(outputA, dim=1)
            outputB = F.softmax(outputB, dim=1)
            output_mix = F.softmax(output_mix, dim=1)

            # Update recorder
            if self.args.classnum > 5:
                prec1_A, prec5_A = accuracy(outputA.data, target1, topk=(1,5))
                prec1_B, prec5_B = accuracy(outputB.data, target1, topk=(1,5))
                prec1_mix, prec5_mix = accuracy(output_mix.data, target1, topk=(1,5))
            else:
                prec1_A = accuracy(outputA.data, target1, topk=(1,))
                prec1_B = accuracy(outputB.data, target1, topk=(1,))
                prec1_mix = accuracy(output_mix.data, target1, topk=(1,))
            top1_A.update(float(prec1_A[0]), input.shape[0])
            top1_B.update(float(prec1_B[0]), input.shape[0])
            top1_mix.update(float(prec1_mix[0]), input.shape[0])
            losses_A.update(float(lossA.data))
            losses_B.update(float(lossA.data))
            pure_ratio_A.update(pure_ratio_1)
            pure_ratio_B.update(pure_ratio_2)
            if pure_ratio_discard_1 >= 0 and pure_ratio_discard_2 >= 0:
                pure_ratio_discard_A.update(pure_ratio_discard_1)
                pure_ratio_discard_B.update(pure_ratio_discard_2)
            if self.args.classnum > 5:
                top5_A.update(float(prec5_A), input.shape[0])
                top5_B.update(float(prec5_B), input.shape[0])
                top5_mix.update(float(prec5_mix), input.shape[0])

            self.optimizerA.zero_grad()
            lossA.backward()
            self.optimizerA.step()

            self.optimizerB.zero_grad()
            lossB.backward()
            self.optimizerB.step()

            # update label distribution
            if epoch >= self.args.stage1 and epoch < self.args.stage2:
                # using select data sample update parameters, other update label only
                yy_A = self.yy
                yy_B = self.yy
                yy_A = torch.tensor(yy_A[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
                yy_B = torch.tensor(yy_B[index, :], dtype=torch.float32, requires_grad=True, device=self.args.device)
                # obtain label distributions (y_hat)
                last_y_var_A = self.softmax(yy_A)
                last_y_var_B = self.softmax(yy_B)
                lossA = self._get_loss(outputA_.detach(), last_y_var_A, loss_type="PENCIL", target_var=target_var)
                lossB = self._get_loss(outputB_.detach(), last_y_var_B, loss_type="PENCIL", target_var=target_var)

                lossA.backward()
                lossB.backward()

                grad = yy_A.grad.data + yy_B.grad.data
                if self.args.mix_grad == 1:
                    yy_A.data.sub_(self.args.lambda1 * grad)
                else:
                    yy_A.data.sub_(self.args.lambda1 * yy_A.grad.data)
                yy_B.data.sub_(self.args.lambda1 * grad)

                if self.args.discard == 1:
                    if self.args.curriculum == 0:
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        self.new_y[index[ind_discard], :] = yy_A.data[ind_discard].cpu().numpy()
                    else:
                        # choose update labels
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        update_stage = self._get_label_update_stage(epoch)
                        subset_labels = self.subset_labels[index[ind_discard]]
                        update_ind = np.where(subset_labels <= update_stage)
                        self.new_y[index[ind_discard[update_ind]], :] = yy_A.data[ind_discard[update_ind]].cpu().numpy()
                else:
                    if self.args.curriculum == 0:
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        self.new_y[index, :] = yy_A.data.cpu().numpy()
                    else:
                        ind_discard = np.unique(np.hstack((ind_A_discard, ind_B_discard)))
                        update_stage = self._get_label_update_stage(epoch)
                        subset_labels = self.subset_labels[index]
                        update_ind = np.where(subset_labels <= update_stage)
                        self.new_y[index[update_ind], :] = yy_A.data[update_ind].cpu().numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print("\rTrain Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                  "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                  "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                  "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                  "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                epoch, self.args.epochs, i, self.train_batch_num,
                batch_time=batch_time, loss_A=losses_A, loss_B=losses_B, top1_A=top1_A, top1_B=top1_B), end='')

        if epoch < self.args.stage2:
            # save y_tilde
            self.yy = self.new_y
            y_file = join(self.args.dir, "y.npy")
            np.save(y_file, self.new_y)
            y_record = join(self.args.dir, "record/y_%03d.npy" % epoch)
            np.save(y_record, self.new_y)

        # check label acc
        label_accu_A, label_n2t_A, label_t2n_A = check_label(self.yy, self.clean_labels, self.noise_or_not, onehotA=True)
        label_accu_B, label_n2t_B, label_t2n_B = check_label(self.yy, self.clean_labels, self.noise_or_not, onehotA=True)
        label_accu_C, label_n2t_C, label_t2n_C = check_label(self.yy, self.clean_labels, self.noise_or_not, onehotA=True)

        if self.args.classnum > 5:
            print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}\tTop5 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg,
                                                                                                 top5_A.avg,
                                                                                                 top5_B.avg))
        else:
            print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))
        print(" * Label accu A: {:.4f}\tB: {:.4f}\tPure ratio A: {:.4f}\tB: {:.4f}".format(label_accu_A, label_accu_B,
                                                                                           pure_ratio_A.avg,
                                                                                           pure_ratio_B.avg))
        print(" * n2t_A: {:.4f}\tn2t_B: {:.4f}\tt2n_A: {:.4f}\tt2n_B: {:.4f}".format(label_n2t_A, label_n2t_B,
                                                                                     label_t2n_A, label_t2n_B))

        self.record_dict['train1']['acc'].append(top1_A.avg)
        self.record_dict['train1']['loss'].append(losses_A.avg)
        self.record_dict['train1']['label_n2t'].append(label_n2t_A)
        self.record_dict['train1']['label_t2n'].append(label_t2n_A)
        self.record_dict['train1']['label_accu'].append(label_accu_A)
        self.record_dict['train1']['pure_ratio'].append(pure_ratio_A.avg)
        self.record_dict['train1']['pure_ratio_discard'].append(pure_ratio_discard_A.avg)
        self.record_dict['train1']['margin_accu'].append(margin_accu.avg)

        self.record_dict['train2']['acc'].append(top1_B.avg)
        self.record_dict['train2']['loss'].append(losses_B.avg)
        self.record_dict['train2']['label_n2t'].append(label_n2t_B)
        self.record_dict['train2']['label_t2n'].append(label_t2n_B)
        self.record_dict['train2']['label_accu'].append(label_accu_B)
        self.record_dict['train2']['pure_ratio'].append(pure_ratio_B.avg)
        self.record_dict['train2']['pure_ratio_discard'].append(pure_ratio_discard_B.avg)
        self.record_dict['train2']['margin_accu'].append(margin_accu.avg)

        self.record_dict['train3']['acc'].append(top1_mix.avg)
        self.record_dict['train3']['label_n2t'].append(label_n2t_C)
        self.record_dict['train3']['label_t2n'].append(label_t2n_C)
        self.record_dict['train3']['label_accu'].append(label_accu_C)
        if self.args.classnum > 5:
            self.record_dict['train1']['acc5'].append(top5_A.avg)
            self.record_dict['train2']['acc5'].append(top5_B.avg)
            self.record_dict['train3']['acc5'].append(top5_mix.avg)
            return top1_A.avg, top1_B.avg, top5_A.avg, top5_B.avg
        return top1_A.avg, top1_B.avg


    def val(self, epoch=0):
        self.modelA.eval()
        self.modelB.eval()

        batch_time = AverageMeter()
        losses_A = AverageMeter()
        top1_A = AverageMeter()
        losses_B = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()
        if self.args.classnum > 5:
            top5_A = AverageMeter()
            top5_B = AverageMeter()
            top5_mix = AverageMeter()

        with torch.no_grad():
            # Validate
            end = time.time()
            for i, (img, label, index) in enumerate(self.valloader):

                img = img.to(self.args.device)
                label = label.to(self.args.device)

                outputA = self.modelA(img)
                lossA = self.criterion(outputA, label)
                outputB = self.modelB(img)
                lossB = self.criterion(outputB, label)
                output_mix = (outputA + outputB) / 2

                outputA = F.softmax(outputA, dim=1)
                outputB = F.softmax(outputB, dim=1)
                output_mix = F.softmax(output_mix, dim=1)

                # Update recorder
                if self.args.classnum > 5:
                    prec1_A, prec5_A = accuracy(outputA.data, label, topk=(1, 5))
                    prec1_B, prec5_B = accuracy(outputB.data, label, topk=(1, 5))
                    prec1_mix, prec5_mix = accuracy(output_mix.data, label, topk=(1, 5))
                else:
                    prec1_A = accuracy(outputA.data, label, topk=(1,))
                    prec1_B = accuracy(outputB.data, label, topk=(1,))
                    prec1_mix = accuracy(output_mix.data, label, topk=(1,))
                top1_A.update(float(prec1_A[0]), img.shape[0])
                losses_A.update(float(lossA.data))
                top1_B.update(float(prec1_B[0]), img.shape[0])
                losses_B.update(float(lossB.data))
                top1_mix.update(float(prec1_mix[0]), img.shape[0])
                if self.args.classnum > 5:
                    top5_A.update(float(prec5_A), img.shape[0])
                    top5_B.update(float(prec5_B), img.shape[0])
                    top5_mix.update(float(prec5_mix), img.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print("\rVal Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  " 
                      "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                      "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                      "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                      "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                    epoch, self.args.epochs, i, self.val_batch_num,
                    batch_time=batch_time, loss_A=losses_A, loss_B=losses_B, top1_A=top1_A, top1_B=top1_B), end='')

            if self.args.classnum > 5:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}\tTop5 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg,
                                                                                                   top5_A.avg, top5_B.avg))
            else:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))

        self.record_dict['val1']['acc'].append(top1_A.avg)
        if self.args.classnum > 5:
            self.record_dict['val1']['acc5'].append(top5_A.avg)
        self.record_dict['val1']['loss'].append(losses_A.avg)

        self.record_dict['val2']['acc'].append(top1_B.avg)
        if self.args.classnum > 5:
            self.record_dict['val2']['acc5'].append(top5_B.avg)
        self.record_dict['val2']['loss'].append(losses_B.avg)

        self.record_dict['val3']['acc'].append(top1_mix.avg)
        if self.args.classnum > 5:
            self.record_dict['val3']['acc5'].append(top5_mix.avg)

        if self.args.classnum > 5:
            return top1_A.avg, top1_B.avg, top5_A.avg, top5_B.avg
        return top1_A.avg, top1_B.avg

    def test(self, epoch=0):
        self.modelA.eval()
        self.modelB.eval()

        batch_time = AverageMeter()
        losses_A = AverageMeter()
        top1_A = AverageMeter()
        losses_B = AverageMeter()
        top1_B = AverageMeter()
        top1_mix = AverageMeter()
        if self.args.classnum > 5:
            top5_A = AverageMeter()
            top5_B = AverageMeter()
            top5_mix = AverageMeter()

        with torch.no_grad():
            # Validate
            end = time.time()
            for i, (img, label, index) in enumerate(self.testloader):

                img = img.to(self.args.device)
                label = label.to(self.args.device)

                outputA = self.modelA(img)
                lossA = self.criterion(outputA, label)
                outputB = self.modelB(img)
                lossB = self.criterion(outputB, label)
                output_mix = (outputA + outputB) / 2

                outputA = F.softmax(outputA, dim=1)
                outputB = F.softmax(outputB, dim=1)
                output_mix = F.softmax(output_mix, dim=1)

                # Update recorder
                if self.args.classnum > 5:
                    prec1_A, prec5_A = accuracy(outputA.data, label, topk=(1, 5))
                    prec1_B, prec5_B = accuracy(outputB.data, label, topk=(1, 5))
                    prec1_mix, prec5_mix = accuracy(output_mix.data, label, topk=(1, 5))
                else:
                    prec1_A = accuracy(outputA.data, label, topk=(1,))
                    prec1_B = accuracy(outputB.data, label, topk=(1,))
                    prec1_mix = accuracy(output_mix.data, label, topk=(1,))
                top1_A.update(float(prec1_A[0]), img.shape[0])
                losses_A.update(float(lossA.data))
                top1_B.update(float(prec1_B[0]), img.shape[0])
                losses_B.update(float(lossB.data))
                top1_mix.update(float(prec1_mix[0]), img.shape[0])
                if self.args.classnum > 5:
                    top5_A.update(float(prec5_A), img.shape[0])
                    top5_B.update(float(prec5_B), img.shape[0])
                    top5_mix.update(float(prec5_mix), img.shape[0])

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print("\rTest Epoch [{0}/{1}]  Batch [{2}/{3}]  "
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                      "LossA {loss_A.val:.3f} ({loss_A.avg:.3f})  "
                      "LossB {loss_B.val:.3f} ({loss_B.avg:.3f})  "
                      "Prec1A {top1_A.val:.3f} ({top1_A.avg:.3f})  "
                      "Prec1B {top1_B.val:.3f} ({top1_B.avg:.3f})".format(
                    epoch, self.args.epochs, i, self.test_batch_num,
                    batch_time=batch_time, loss_A=losses_A, loss_B=losses_B, top1_A=top1_A, top1_B=top1_B), end='')

            if self.args.classnum > 5:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}\tTop5 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg,
                                                                                                   top5_A.avg, top5_B.avg))
            else:
                print("\n * Top1 acc:\tA: {:.3f}\tB: {:.3f}".format(top1_A.avg, top1_B.avg))

        self.record_dict['test1']['acc'].append(top1_A.avg)
        if self.args.classnum > 5:
            self.record_dict['test1']['acc5'].append(top5_A.avg)
        self.record_dict['test1']['loss'].append(losses_A.avg)

        self.record_dict['test2']['acc'].append(top1_B.avg)
        if self.args.classnum > 5:
            self.record_dict['test2']['acc5'].append(top5_B.avg)
        self.record_dict['test2']['loss'].append(losses_B.avg)

        self.record_dict['test3']['acc'].append(top1_mix.avg)
        if self.args.classnum > 5:
            self.record_dict['test3']['acc5'].append(top5_mix.avg)

        if self.args.classnum > 5:
            return top1_A.avg, top1_B.avg, top5_A.avg, top5_B.avg
        return top1_A.avg, top1_B.avg


def save_checkpoint(state, is_best, filename='', modelbest=''):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, modelbest)
        print("Saving best model at epoch {}".format(state['epoch']))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    args = get_args()
    trainer = CoCorrecting(args=args)
    trainer.training()