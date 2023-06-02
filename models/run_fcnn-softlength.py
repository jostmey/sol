#!/usr/bin/env python3
##########################################################################################
# Author: Jared L. Ostmeyer
##########################################################################################

##########################################################################################
# Libraries
##########################################################################################

import argparse
import torchvision
import torch
from SoftLength import *

##########################################################################################
# Arguments
##########################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--batch', help='Batch size', type=int, default=1024)
parser.add_argument('--step', help='Step size', type=float, default=0.001)
parser.add_argument('--epochs', help='Number of passes over the dataset', type=int, default=1024)
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

##########################################################################################
# Load data
##########################################################################################

transform = torchvision.transforms.Compose(
  [
    torchvision.transforms.ToTensor(),
    torch.nn.Flatten(start_dim=0, end_dim=2)
  ]
)

dataset = torchvision.datasets.MNIST(root='../datasets', train=True, download=True, transform=transform)
dataset_test = torchvision.datasets.MNIST(root='../datasets', train=False, download=True, transform=transform)

num = len(dataset)
num_train = int(5/6*num)
num_val = num-num_train

dataset_train, dataset_val = torch.utils.data.random_split(dataset, [ num_train, num_val ], generator=generator)

sampler_train = torch.utils.data.RandomSampler(dataset_train, replacement=True, generator=generator)
sampler_val = torch.utils.data.SequentialSampler(dataset_val)
sampler_test = torch.utils.data.SequentialSampler(dataset_test)

loader_train = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=args.batch, sampler=sampler_train)
loader_val = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=args.batch, sampler=sampler_val)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args.batch, sampler=sampler_test)

##########################################################################################
# Model
##########################################################################################

class Model(torch.nn.Module):
  def __init__(self, num_inputs, num_outputs, num_hiddens=512, p=0.5, **kwargs):
    super().__init__(**kwargs)
    self.norm0 = torch.nn.BatchNorm1d(num_inputs)

    self.layer1 = torch.nn.Sequential(
      torch.nn.Linear(num_inputs, num_hiddens),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=p),
      torch.nn.BatchNorm1d(num_hiddens)
    )
    self.output1 = torch.nn.Linear(num_hiddens, num_outputs)

    self.layer2 = torch.nn.Sequential(
      torch.nn.Linear(num_hiddens, num_hiddens),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=p),
      torch.nn.BatchNorm1d(num_hiddens)
    )
    self.output2 = torch.nn.Linear(num_hiddens, num_outputs)

    self.layer3 = torch.nn.Sequential(
      torch.nn.Linear(num_hiddens, num_hiddens),
      torch.nn.ReLU(),
      torch.nn.Dropout(p=p),
      torch.nn.BatchNorm1d(num_hiddens)
    )
    self.output3 = torch.nn.Linear(num_hiddens, num_outputs)

    self.softlength = SoftLength(width=4096)
  def forward(self, x):
    n0 = self.norm0(x)

    l1 = self.layer1(n0)
    o1 = self.output0(l1)

    l2 = self.layer2(l1)
    o2 = self.output1(l2)

    l3 = self.layer3(l2)
    o3 = self.output2(l3)

    o = torch.stack([ o1, o2, o3 ], axis=1)
    p = self.softlength(o)
    return p

##########################################################################################
# Instantiate model, optimizer, and metrics
##########################################################################################

model = Model(28**2, 10).to(device)

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