# -*- coding: utf-8 -*-
# @Author: chengdlin2
# @Date:   2020-02-23 19:19:29
# @Last Modified by:   chengdlin2
# @Last Modified time: 2020-03-07 18:52:30


# use MINIST Classifier and Sine Regressor as basic model to test maml
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

class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    
    def __init__(self, amplitude, phase, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.xmin = xmin
        self.xmax = xmax
        
    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        
        return self.amplitude * np.sin(self.phase + x)
        
    def sample_data(self, size=1, split='s'):
        """
        Sample data from this task.
        
        returns: 
            x: the feature vector of length size
            y: the target vector of length size
        """
        
        x = np.random.uniform(self.xmin, self.xmax, size)
        y = self.true_function(x)
        
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1).cuda()
        y = torch.tensor(y, dtype=torch.float).unsqueeze(1).cuda()
        
        return x, y

class Sine_Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """
    
    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max
        
    def sample_task(self):
        """
        Sample from the task distribution.
        
        returns:
            Sine_Task object
        """
        amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        return Sine_Task(amplitude, phase, self.x_min, self.x_max)

# basic MLP model for testing SINE regression 
class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
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
    # training and model saving
    
    tasks = Sine_Task_Distribution(0.1, 5, 0, np.pi, -5, 5) # tasks is meta dataset including a set of sine functions with different amplititude and phase
    maml = MAML(Regressor().cuda(), tasks, inner_lr=0.01, meta_lr=0.005)
    maml.main_loop(num_iterations=15000)
    PATH = './checkpoint.pth'
    torch.save(maml.model.state_dict(), PATH)

    # Testing and visualize result
    model = Regressor()
    model.load_state_dict(torch.load(PATH))
    model.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.01)

    # fine tuning
    model.train()
    test_task = tasks.sample_task()
    X, y = test_task.sample_data(50)
    fine_steps = 10
    x_axis = np.linspace(-5, 5, 1000, dtype=np.float32)
    outputs = {}
    outputs['init'] = model(torch.from_numpy(x_axis).view(-1,1).cuda()).detach().cpu().numpy()
    losses = []
    for step in range(fine_steps):
        loss = criterion(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        outputs[step] = model(torch.from_numpy(x_axis).view(-1,1).cuda()).detach().cpu().numpy()

    # visualization
    plt.figure()
    plt.subplot(2, 2, 1)
    for i in [1,9]:
        plt.plot(x_axis, outputs[i], '-.' if i == 1 else '-', color=(0.5, 0, 0, 1),
                 label='model after {} steps'.format(i))
    plt.title("Maml Sine Result")
    plt.plot(x_axis, outputs['init'], ':', color=(0.7, 0, 0, 1), label='initial weights')
    plt.plot(x_axis, test_task.true_function(x_axis), '-', color=(0, 0, 1, 0.5), label='true function')
    X = X.to(torch.device('cpu'))
    y = y.to(torch.device('cpu'))
    plt.scatter(X, y, label='data')
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 2)
    plt.title("Fine Tune Loss")
    plt.plot(losses)
    plt.subplot(2,2,3)
    plt.title("Meta Train Loss")
    plt.plot(maml.plot_losses)
    plt.show()