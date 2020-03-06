# -*- coding: utf-8 -*-
# @Author: chengdlin2
# @Date:   2020-03-06 11:57:20
# @Last Modified by:   chengdlin2
# @Last Modified time: 2020-03-06 22:18:39

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
    def __init__(self, support_loader, query_loader):
        self.query = iter(query_loader)
        self.support = iter(support_loader)
        
    def sample_data(self, split='s'):
        """
        Sample data from this task.
        returns: 
            x: the input image
            y: the class label
        """
        if split == 's':
            return next(self.support)
        if split == 'q':
            return next(self.query)

class Omniglot_Task_Distribution():
    """
    The task distribution for omniglot classification tasks for MAML
    """
    def __init__(self, dataset, num_per_class):
        # each class of Omniglot has 20 instances, thus len(dataset)/20 is the number of classes included
        self.dataset = dataset
        self.num_per_class = num_per_class
        self.N_C = int(len(self.dataset)/self.num_per_class)
    def sample_task(self, N, K, M):
        """
        Sample a N way K shot classification problem from the task distribution.
        returns:
            Omniglot_Task object
        """
        # random choice
        class_index = torch.randperm(self.N_C).tolist()[:N]
        print(class_index)
        indices = [(torch.randperm(self.num_per_class) + c_idx*self.num_per_class ).tolist() for c_idx in class_index]
        support_loader = torch.utils.data.DataLoader(self.dataset, batch_size = K*N, 
                            sampler = torch.utils.data.SubsetRandomSampler(sum([i[:K] for i in indices], [])))
        query_loader = torch.utils.data.DataLoader(self.dataset, batch_size = M*N, 
                            sampler = torch.utils.data.SubsetRandomSampler(sum([i[K:] for i in indices], [])))

        return Omniglot_Task(support_loader, query_loader)

# basic MLP model for testing SINE regression 
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('l1', nn.Linear(1,40)),
            ('relu1', nn.ReLU()),
            ('l2', nn.Linear(40,40)),
            ('relu2', nn.ReLU()),
            ('l3', nn.Linear(40,1))
        ]))
        
    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    trans = transforms.Compose([transforms.Resize((80, 80)),transforms.ToTensor()])
    tasks = Omniglot_Task_Distribution(datasets.Omniglot('./Omniglot/', transform=trans), 20)
    t = datasets.Omniglot('./Omniglot/', background=False)
    task = tasks.sample_task(5, 5, 15)
    print(task.sample_data()[1])
    print(task.sample_data('q')[1])
