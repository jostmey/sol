#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
# Date Started: 2023-05-26
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import torchvision
import torch
from SoftLength_v3 import *

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--batch', help='Batch size', type=int, default=512)
parser.add_argument('--step', help='Step size', type=float, default=0.0001)
parser.add_argument('--epochs', help='Number of passes over the dataset', type=int, default=256)
parser.add_argument('--device', help='Examples are cuda:0 or cpu', type=str, default='cpu')
parser.add_argument('--output', help='Path for saving the model', type=str, default=None)
args = parser.parse_args()

##########################################################################################
# Settings
##########################################################################################

device = torch.device(args.device)

seed = 90843
generator = torch.Generator()
generator.manual_seed(seed)

##########################################################################################
# Load data
##########################################################################################

transform = torchvision.transforms.Compose(
  [
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.Resize([ 56, 56 ]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation([ -7, 7 ]),
    torchvision.transforms.RandomAffine(0, shear=10, scale=[ 0.8, 1.2 ]),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
      mean=[ 0.4914, 0.4822, 0.4465 ],
      std=[ 0.2023, 0.1994, 0.2010 ]
    ),
  ]
)
transform_valtest = torchvision.transforms.Compose(
  [
    torchvision.transforms.Resize([ 56, 56 ]),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
      mean=[ 0.4914, 0.4822, 0.4465 ],
      std=[ 0.2023, 0.1994, 0.2010 ]
    ),
  ]
)

dataset_train = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform)
dataset_val = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform_valtest)
dataset_test = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform_valtest)

num = len(dataset_train)
num_train = int(0.9*num)

indices = torch.randperm(num, generator=generator)
indices_train, indices_val = indices[:num_train], indices[num_train:]

dataset_train = torch.utils.data.Subset(dataset_train, indices_train)
dataset_val = torch.utils.data.Subset(dataset_val, indices_val)

sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True, generator=generator)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
sampler_test = torch.utils.data.SequentialSampler(dataset_test)

loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch, sampler=sampler_train)
loader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.batch, sampler=sampler_val)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch, sampler=sampler_test)

##########################################################################################
# Model
##########################################################################################

class ResidualBlock(torch.nn.Module):
  def __init__(self, in_channels, out_channels, stride=1, downsample=None, **kwargs):
    super().__init__(**kwargs)
    self.conv1 = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
      torch.nn.BatchNorm2d(out_channels),
      torch.nn.ReLU()
    )
    self.conv2 = torch.nn.Sequential(
      torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(out_channels)
    )
    self.downsample = downsample
    self.relu = torch.nn.ReLU()
  def forward(self, x):
    residual = x
    out = self.conv1(x)
    out = self.conv2(out)
    if self.downsample:
      residual = self.downsample(x)
    out += residual
    out = self.relu(out)
    return out

class ResNet(torch.nn.Module):
  def __init__(self, block, layers, num_classes=10, **kwargs):
    super().__init__(**kwargs)

    self.input = torch.nn.Sequential(
      torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
      torch.nn.BatchNorm2d(64),
      torch.nn.ReLU(),
    )

    blocks = []
    outputs = []

    blocks.append(block(64, 64, 1))
    outputs.append(
      torch.nn.Sequential(
        torch.nn.AvgPool2d(7, stride=1),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(64*50*50),
        torch.nn.Linear(64*50*50, num_classes)
      )
    )
    for i in range(1, layers[0]):
      blocks.append(block(64, 64))
      outputs.append(
        torch.nn.Sequential(
          torch.nn.AvgPool2d(7, stride=1),
          torch.nn.Flatten(),
          torch.nn.BatchNorm1d(64*50*50),
          torch.nn.Linear(64*50*50, num_classes)
        )
      )

    downsample = torch.nn.Sequential(
      torch.nn.Conv2d(64, 128, kernel_size=1, stride=2),
      torch.nn.BatchNorm2d(128)
    )
    blocks.append(block(64, 128, 2, downsample=downsample))
    outputs.append(
      torch.nn.Sequential(
        torch.nn.AvgPool2d(7, stride=1),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(128*22*22),
        torch.nn.Linear(128*22*22, num_classes)
      )
    )
    for i in range(1, layers[1]):
      blocks.append(block(128, 128))
      outputs.append(
        torch.nn.Sequential(
          torch.nn.AvgPool2d(7, stride=1),
          torch.nn.Flatten(),
          torch.nn.BatchNorm1d(128*22*22),
          torch.nn.Linear(128*22*22, num_classes)
        )
      )

    downsample = torch.nn.Sequential(
      torch.nn.Conv2d(128, 256, kernel_size=1, stride=2),
      torch.nn.BatchNorm2d(256)
    )
    blocks.append(block(128, 256, 2, downsample=downsample))
    outputs.append(
      torch.nn.Sequential(
        torch.nn.AvgPool2d(7, stride=1),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(256*8*8),
        torch.nn.Linear(256*8*8, num_classes)
      )
    )
    for i in range(1, layers[2]):
      blocks.append(block(256, 256))
      outputs.append(
        torch.nn.Sequential(
          torch.nn.AvgPool2d(7, stride=1),
          torch.nn.Flatten(),
          torch.nn.BatchNorm1d(256*8*8),
          torch.nn.Linear(256*8*8, num_classes)
        )
      )

    downsample = torch.nn.Sequential(
      torch.nn.Conv2d(256, 512, kernel_size=1, stride=2),
      torch.nn.BatchNorm2d(512)
    )
    blocks.append(block(256, 512, 2, downsample=downsample))
    outputs.append(
      torch.nn.Sequential(
        torch.nn.AvgPool2d(7, stride=1),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(512),
        torch.nn.Linear(512, num_classes)
      )
    )
    for i in range(1, layers[3]):
      blocks.append(block(512, 512))
      outputs.append(
        torch.nn.Sequential(
          torch.nn.AvgPool2d(7, stride=1),
          torch.nn.Flatten(),
          torch.nn.BatchNorm1d(512),
          torch.nn.Linear(512, num_classes)
        )
      )

    self.blocks = torch.nn.ModuleList(blocks)
    self.outputs = torch.nn.ModuleList(outputs)

    self.softlength = SoftLength(width=4096)

  def forward(self, x):
    x = self.input(x)
    o = []
    for block, output in zip(self.blocks, self.outputs):
      x = block(x)
      o.append(output(x))
    o = torch.stack(o, axis=1)
    p = self.softlength(o)
    return p

##########################################################################################
# Instantiate model, optimizer, and metrics
##########################################################################################

model = ResNet(ResidualBlock, [ 3, 4, 6, 3 ]).to(device)
softmax = torch.nn.Softmax(dim=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.step)

loss = SoftLengthLoss(epsilon=1.0e-3).to(device)
accuracy = SoftLengthAccuracy().to(device)

##########################################################################################
# Run the model
##########################################################################################

i_best = -1
e_best = 1.0e8
a_best = 0.0
state_best = {}

for i in range(args.epochs):

  e_train = 0.0
  a_train = 0.0
  model.train()
  for xs_batch, ys_batch in iter(loader_train):
    xs_batch = xs_batch.to(device)
    ys_batch = ys_batch.to(device)
    ps_batch = model(xs_batch)
    e_batch = loss(ps_batch, ys_batch)
    a_batch = accuracy(ps_batch, ys_batch)
    optimizer.zero_grad()
    e_batch.backward()
    optimizer.step()
    fraction = float(ys_batch.shape[0])/float(len(dataset_train))
    e_train += fraction*e_batch.detach()
    a_train += fraction*a_batch.detach()

  e_val = 0.0
  a_val = 0.0
  model.eval()
  with torch.no_grad():
    for xs_batch, ys_batch in iter(loader_val):
      xs_batch = xs_batch.to(device)
      ys_batch = ys_batch.to(device)
      ps_batch = model(xs_batch)
      e_batch = loss(ps_batch, ys_batch)
      a_batch = accuracy(ps_batch, ys_batch)
      fraction = float(ys_batch.shape[0])/float(len(dataset_val))
      e_val += fraction*e_batch.detach()
      a_val += fraction*a_batch.detach()
    if e_val < e_best:
      i_best = i
      e_best = e_val
      a_best = a_val
      state_best = model.state_dict()
  print(
    'i:', i,
    'e_train:', float(e_train)/0.693, 'a_train:', 100.0*float(a_train),
    'e_val:', float(e_val)/0.693, 'a_val:', 100.0*float(a_val),
    sep='\t', flush=True
  )

e_test = 0.0
a_test = 0.0
model.load_state_dict(state_best)
model.eval()
with torch.no_grad():
  for xs_batch, ys_batch in iter(loader_test):
    xs_batch = xs_batch.to(device)
    ys_batch = ys_batch.to(device)
    ps_batch = model(xs_batch)
    e_batch = loss(ps_batch, ys_batch)
    a_batch = accuracy(ps_batch, ys_batch)
    fraction = float(ys_batch.shape[0])/float(len(dataset_test))
    e_test += fraction*e_batch.detach()
    a_test += fraction*a_batch.detach()

print(
  'i_best:', i_best,
  'e_val:', float(e_best)/0.693,
  'a_val:', 100.0*float(a_best),
  'e_test:', float(e_test)/0.693,
  'a_test:', 100.0*float(a_test),
  sep='\t', flush=True
)

##########################################################################################
# Save the model
##########################################################################################

if args.output is not None:
  torch.save(model.state_dict(), args.output)
