import torch
import os
import numpy as np
import torch_geometric as tg


class RandomScale():
    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, data):
        scale = torch.rand(1).item() * (self.scale_high - self.scale_low) + self.scale_low
        data.pos = data.pos * scale
        return data
    

class RandomDropColor():
    def __init__(self, p=0.8, color_augment=0.0):
        self.p = p
        self.color_augment = color_augment
    
    def __call__(self, data):
        if np.random.rand() > self.p:
            data.x *= self.color_augment
        return data