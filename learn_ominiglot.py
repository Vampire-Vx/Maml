# -*- coding: utf-8 -*-
# @Author: chengdlin2
# @Date:   2020-03-06 11:57:20
# @Last Modified by:   chengdlin2
# @Last Modified time: 2020-03-07 19:04:36

# reproduce Maml training on Omniglot dataset
# Ominiglot: 1623 handwritten characters from 50 different alphabets, each has 20 instances

import PIL
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from collections import OrderedDict
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from maml import MAML
class Omniglot_Task():
    """
    A task contain N way K shot classification Support set & Query Set
    """
    def __init__(self, support_loader, query_loader, class_index):
        self.query = iter(query_loader)
        self.support = iter(support_loader)
        self.index_map = {value:index for index, value in enumerate(class_index)}
    def sample_data(self, size=1, split='s'):
        """
        Sample data from this task.
        returns: 
            x: the input image
            y: the class label
        """
        if split == 's':
            x, y = next(self.support)       
        if split == 'q':
            x, y = next(self.query)
        y = [self.index_map[i] for i in y.numpy()]
        return x.detach().cuda(), torch.tensor(y).cuda()

class Omniglot_Task_Distribution():
    """
    The task distribution for omniglot classification tasks for MAML
    """
    def __init__(self, dataset, num_per_class):
        # each class of Omniglot has 20 instances, thus len(dataset)/20 is the number of classes included
        self.dataset = dataset
        self.num_per_class = num_per_class
        self.N_C = int(len(self.dataset)/self.num_per_class)
    def sample_task(self, N=5, K=5, M=15):
        """
        Sample a N way K shot classification problem from the task distribution.
        returns:
            Omniglot_Task object
        """
        # random choice
        class_index = torch.randperm(self.N_C).tolist()[:N]
        indices = [(torch.randperm(self.num_per_class) + c_idx*self.num_per_class ).tolist() for c_idx in class_index]
        support_loader = torch.utils.data.DataLoader(self.dataset, batch_size = K*N, 
                            sampler = torch.utils.data.SubsetRandomSampler(sum([i[:K] for i in indices], [])))
        query_loader = torch.utils.data.DataLoader(self.dataset, batch_size = M*N, 
                            sampler = torch.utils.data.SubsetRandomSampler(sum([i[K:] for i in indices], [])))

        return Omniglot_Task(support_loader, query_loader, class_index)

# basic MLP model for testing SINE regression 
class Classifier(nn.Module):
    def __init__(self, N):
        super(Classifier, self).__init__()
        self.conv1 = self.conv_block(1,64)
        self.conv2 = self.conv_block(64,64)
        self.conv3 = self.conv_block(64,64)
        self.conv4 = self.conv_block(64,64)
        self.logits = nn.Linear(64, N)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding = 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)

            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = x.view(x.size(0), -1)

        return self.logits(x)


if __name__ == "__main__":

    trans = transforms.Compose([transforms.Resize((28, 28)),transforms.ToTensor()])
    tasks = Omniglot_Task_Distribution(datasets.Omniglot('./Omniglot/', transform=trans), 20)
    N, K = 5, 5
    task = tasks.sample_task(N, K, 15)
    meta_model = Classifier(N)
    maml = MAML(meta_model.cuda(), tasks, 
        inner_lr=0.01, meta_lr=0.001, K=10, 
        inner_steps=1, tasks_per_meta_batch=32, criterion=nn.CrossEntropyLoss())
    maml.main_loop(num_iterations=100)

