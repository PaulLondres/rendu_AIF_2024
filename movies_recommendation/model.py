#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:39:07 2023

@author: deltort
"""

import torchvision.models as models
import torch
import torch.nn as nn


mobilenet = models.mobilenet_v3_small(weights='DEFAULT')
features = mobilenet.features
flatten = nn.Flatten()

# model =torch.nn.Sequential(features, torch.nn.AdaptiveAvgPool2d(1),flatten)
model1 =torch.nn.Sequential(features, torch.nn.AdaptiveAvgPool2d(1))
model = nn.Sequential(model1,flatten)
model.eval()
# model.train()