##########################################################################################
# Author: Jared L. Ostmeyer
##########################################################################################

import torch

class SoftLength(torch.nn.Module):
  def __init__(self, width, **kwargs):
    super().__init__(**kwargs)
    self.linear_1 = torch.nn.Linear(1, width)
    self.relu = torch.nn.ReLU()
    self.linear_2 = torch.nn.Linear(width, 1)

    self.softmax2 = torch.nn.Softmax(dim=1)
    self.sigmoid = torch.nn.Sigmoid()
    self.softmax3 = torch.nn.Softmax(dim=2)
  def forward(self, logits):
    batch_size, num_steps, num_outputs = logits.shape

    logits_trim = logits[:,:-1,:]
    logits_flat = torch.reshape(logits_trim, [ -1, num_outputs ])
    probabilities_flat = self.softmax2(logits_flat)
    entropies_flat = -probabilities_flat*(logits_flat-torch.logsumexp(logits_flat, dim=1, keepdim=True))
    entropy_flat = torch.sum(entropies_flat, dim=1, keepdim=True)
    entropy_max = torch.log(torch.tensor(logits.shape[1], dtype=logits.dtype))
    norm_flat = 2.0*entropy_flat/entropy_max-1.0

    linears_flat = self.linear_1(norm_flat)
    hiddens_flat = self.relu(linears_flat)
    norms_flat= 2.0*hiddens_flat
    linears_flat = self.linear_2(norms_flat)
    linears_trim = torch.reshape(linears_flat, [ batch_size, num_steps-1, 1 ])

    ns = torch.reshape(
      torch.range(num_steps-1,1,-1, dtype=logits.dtype).to(logits.device),
      [ 1, num_steps-1, 1 ]
    )
    flags_trim = torch.sigmoid(linears_trim-torch.log(ns))
    zeros = torch.zeros([ batch_size, 1, 1 ], dtype=logits.dtype).to(logits.device) # Active flag for last step
    ones = torch.ones([ batch_size, 1, 1 ], dtype=logits.dtype).to(logits.device) # Active flag for last step
    flags = torch.concat([ zeros, flags_trim, ones ], axis=1)

    probabilities = self.softmax3(logits)
    residuals = torch.cumprod(1.0-flags[:,:-1], dim=1)
    frequencies = probabilities*flags[:,1:]*residuals

    return frequencies

class SoftLengthLoss(torch.nn.Module):
  def __init__(self, epsilon=1.0e-3, dtype=torch.float32, **kwargs):
    super().__init__(**kwargs)
    self.epsilon = epsilon
    self.loss = torch.nn.CrossEntropyLoss(reduction='none')
  def forward(self, frequencies, targets):
    batch_size, num_steps, num_categories = frequencies.shape
    dtype = frequencies.dtype

    indices = torch.eye(num_categories, dtype=dtype).unsqueeze(0).tile([ num_steps, 1, 1 ]).to(frequencies.device)
    steps = torch.range(1, num_steps, dtype=dtype).reshape([ num_steps, 1, 1 ]).to(frequencies.device)
    probabilities = ((1-indices)*self.epsilon*steps+indices)/((num_categories-1)*self.epsilon*steps+1)

    probabilities_tile = probabilities.type(dtype).unsqueeze(0).tile([ batch_size, 1, 1, 1 ])
    targets_tile = targets.reshape([ batch_size, 1, 1 ]).tile([ 1, num_steps, num_categories ])

    probabilities_flat = probabilities_tile.reshape([ batch_size*num_steps*num_categories, num_categories ])
    targets_flat = targets_tile.reshape([ batch_size*num_steps*num_categories ])
    losses_flat = self.loss(torch.log(probabilities_flat), targets_flat)
    losses = losses_flat.reshape([ batch_size, num_steps, num_categories])

    loss = torch.mean(torch.sum(frequencies*losses, axis=[1, 2]))
    return loss

class SoftLengthAccuracy(torch.nn.Module):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
  def forward(self, frequencies, targets):
    batch_size, num_steps, num_categories = frequencies.shape
    dtype = frequencies.dtype

    frequencies = torch.reshape(frequencies, [ batch_size, num_steps*num_categories ])
    guesses = torch.argmax(frequencies, dim=1)%num_categories

    matches = ( guesses == targets ).type(dtype)

    accuracy = torch.mean(matches)
    return accuracy
