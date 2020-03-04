# -*- coding: utf-8 -*-
# @Author: chengdlin2
# @Date:   2019-10-20 14:45:59
# @Last Modified by:   chengdlin2
# @Last Modified time: 2020-03-03 21:57:38
"""Meta Learner Implementation of Maml
   1st order approximation
   todo: 2nd order version extension
   Reference:
       https://github.com/dragen1860/MAML-Pytorch
       https://towardsdatascience.com/advances-in-few-shot-learning-reproducing-results-in-pytorch-aba70dee541d
       https://github.com/vmikulik/maml-pytorch/blob/master/MAML-Sines.ipynb
"""
import torch
import torch.optim
import torch.nn as nn
import numpy as np
import copy
from collections import OrderedDict

def replace_grad(parameter_gradients, parameter_name):
    # hook function should not modify its argument
    def replace_grad_(grad):
        return parameter_gradients[parameter_name]
    # return/bind a hook function for each parameter
    return replace_grad_

class MAML():
    def __init__(self, model, tasks, inner_lr, meta_lr, K=10, inner_steps=1, tasks_per_meta_batch=1000):
        
        # important objects
        self.tasks = tasks
        self.model = model
        self.criterion = nn.MSELoss()
        self.meta_optimizer = torch.optim.SGD(model.parameters(), meta_lr)
        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.K = K # inner loop samples
        self.inner_steps = inner_steps # with the current design of MAML, >1 is unlikely to work well 
        self.tasks_per_meta_batch = tasks_per_meta_batch 
        self.plot_losses = []
    def inner_loop(self, task):
        """inner loop of MAML
           For fist order version, create a new model and copy weights from old model for simplicity
           Sample K for update fast model, Sample K for compute meta loss for a task
           copy method: load_state_dict or deep copy
        """
        # get K samples for a specific task from trainning data
        X_train, y_train = task.sample_data(self.K)
        # instantiate a completely new model using deepcopy
        # if using 2nd order approximation, fuctional_forward may needed
        fast_model = copy.deepcopy(self.model)
        fast_optim = torch.optim.SGD(fast_model.parameters(), self.inner_lr)
        # training on data sampled from task on the new model
        for step in range(self.inner_steps):
            output = fast_model(X_train)
            loss = self.criterion(output, y_train)
            fast_optim.zero_grad()
            loss.backward()
            # update 
            fast_optim.step()
        # sample new data for meta update
        X_val, y_val = task.sample_data(self.K)
        output = fast_model(X_val)
        # loss is for 2nd order version of MAML
        loss = self.criterion(output, y_val)
        # compute grad and store grads for each weight
        fast_weights = OrderedDict(fast_model.named_parameters())
        gradients = torch.autograd.grad(loss, fast_weights.values())
        named_grads = {name: g for((name, _),g) in zip(fast_weights.items(),gradients)}
        return loss, named_grads
        # sample new data for meta-update and compute loss
    
    def main_loop(self, num_iterations):
        epoch_loss = 0
        epoch_size = 10
        for iteration in range(1, num_iterations+1):
            # compute meta loss
            task_losses = []
            task_gradients = []
            for i in range(self.tasks_per_meta_batch):
            # outer loop of MAML
                task = self.tasks.sample_task()
                meta_loss, named_grads = self.inner_loop(task)
                task_losses.append(meta_loss)
                task_gradients.append(named_grads)
            # sum over tasks for parameters with the same name
            sum_task_gradients = {k: torch.stack([grad[k] for grad in task_gradients]).mean(dim=0) for k in task_gradients[0].keys()}
            
            # register hooks for each parameter in original model
            hooks = []
            for name, param in self.model.named_parameters():
                hooks.append(
                        param.register_hook(replace_grad(sum_task_gradients, name))
                )
                
            self.model.train()
            self.meta_optimizer.zero_grad()
            # Dummy pass to build computation graph
            # Replace mean task gradients for dummy gradients
            # todo: create dummy inputs
            X_dummy, y_dummy = self.tasks.sample_task().sample_data()
            output_dummy = self.model(X_dummy)
            loss = self.criterion(output_dummy, y_dummy)
            loss.backward()
            self.meta_optimizer.step()
            # remember to remove hooks to release memory
            for h in hooks:
                h.remove()
            epoch_loss += torch.mean(torch.stack(task_losses)).item()
            if iteration % epoch_size == 0:
                self.plot_losses.append(epoch_loss/epoch_size)
                print("{}/{}. loss : {}".format(iteration, num_iterations, epoch_loss/epoch_size))
                epoch_loss = 0
    