from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import unicodedata
import string

useGPU = torch.cuda.is_available()
print("cuda" if useGPU else "cpu")
device = torch.device("cuda" if useGPU else "cpu")

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters) + 1 # include padding

def findFiles(path): return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
  return ''.join(
    c for c in unicodedata.normalize('NFD', s)
    if unicodedata.category(c) != 'Mn'
    and c in all_letters
  )
# Read a file and split into lines
def readLines(filename):
  lines = open(filename, encoding='utf-8').read().strip().split('\n')
  return [unicodeToAscii(line) for line in lines]


class NameCountryDataset(Dataset):
  def __init__(self, root_dir="data/names", transform=None):
    self.root_dir = root_dir
    self.transform = transform

    self.data = []
    self.all_categories = []
    self.name_len = 0
    
    datapath = os.path.join(self.root_dir, '*.txt')

    for filename in findFiles(datapath):
      category = os.path.splitext(os.path.basename(filename))[0]
      self.all_categories.append(category)
      lines = readLines(filename)

      for name in lines:
        self.data.append({'name': name, 'label': category})
        if (len(name) > self.name_len):
          self.name_len = len(name)

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    
    sample = self.data[idx]
    
    if self.transform:
      sample = self.transform(sample)
    
    return sample

class Pad(object):
  def __init__(self, output_len):
    assert isinstance(output_len, int)
    self.output_len = output_len
  
  def __call__(self, sample):
    name, label = sample['name'], sample['label']
    name += "0" * (self.output_len - len(name))
    return {"name": name, "label": label}

mydataset = NameCountryDataset()
print(mydataset.all_categories)
# print(mydataset.name_len)

len_data = len(mydataset)

print("data size: ", len_data)

n_train_data = int(len_data * 0.9)
n_val_data = len_data - n_train_data
train_dataset, val_dataset = random_split(mydataset, [n_train_data, n_val_data])

batch_size = 1

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

def letterToIndex(letter):
  if (letter == '0'):
    return 0
  return all_letters.find(letter) + 1

# one-hot vector
def lineToTensor(line):
  tensor = torch.zeros(len(line), 1, n_letters)
  for li, letter in enumerate(line):
    tensor[li][0][letterToIndex(letter)] = 1
  return tensor

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
n_categories = 18
model = RNN(n_letters, n_hidden, n_categories)
# optimizer는 없어도 되는건가???

if useGPU:
  model = model.to(device)

def categoryFromOutput(output, dataset):
  top_n, top_i = output_topk(1)
  category_i = top_i[0].item()
  return dataset.all_categories[category_i], category_i

criterion = nn.NLLLoss()
learning_rate = 0.005

def train(category_tensor, line_tensor):
  hidden = model.initHidden()

  for i in range(line_tensor.size()[0]):
    output, hidden = rnn(line_tensor[i], hidden)
  
  loss = criterion(output, category_tensor)
  loss.backward()

  for p in rnn.parameters():
    p.data.add_(p.grad.data, alpha=-learning_rate)
  
  return output, loss.item()

import time
import math

n_iters = 1
print_every = 10
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#  numpy arrays to save loss & accuracy from each epoch
train_loss_iter = np.zeros(n_iters, dtype=float)  # Temporary numpy array to save loss for each epoch
valid_loss_iter = np.zeros(n_iters, dtype=float)
train_accuracy_iter = np.zeros(n_iters, dtype=float)  # Temporary numpy array to save accuracy for each epoch
valid_accuracy_iter = np.zeros(n_iters, dtype=float)

for iter in range(n_iters):

  # Training
  total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0

  # for each mini-batch
  for batch_idx, sample_batched in enumerate(train_loader):
    # print(sample_batched)
    names = sample_batched["name"][0]
    labels = sample_batched["label"][0]

    # batch를 어떻게 처리해야할까..?
    category_tensor = torch.tensor([mydataset.all_categories.index(labels)], dtype=torch.long)
    line_tensor = lineToTensor(names)
    hidden = model.initHidden()

    if useGPU:
      category_tensor = category_tensor.cuda()
      line_tensor = line_tensor.cuda()
      hidden = hidden.cuda()

    for i in range(line_tensor.size()[0]):
      pred, hidden = model(line_tensor[i], hidden)

    optimizer.zero_grad()
    loss = criterion(pred, category_tensor)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() # accumulate loss

    # x, target이 뭘까
    total_cnt += batch_size # accumulate the number of data
  #   correct_cnt += (thresholding(prediction) == target.data).sum().item()  # accumulate the number of correct predictions

  # accuracy = correct_cnt * 1.0 / total_cnt  # calculate accuracy  (#accumulated-correct-prediction/#accumulated-data)
  print("total_loss: ", total_loss)
  print("total cnt: ", total_cnt)
  train_loss_iter[iter] = total_loss / total_cnt # calculate and save loss (#accumulated-loss/#accumulated-data)
  # train_accuracy_iter[iter] = accuracy  # save accuracy

  # Validation
  # total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0
  # for batch_idx, sample_batched in enumerate(valid_loader):
  #   with torch.no_grad():
  #     if useGPU:

    # output, loss = train(category_tensor, line_tensor)
    # current_loss += loss

  # Print iter number, loss, name and guess
  # if iter % print_every == 0:
  print(f"[{iter}/{n_iters}] Train Loss : {train_loss_iter[iter]:.4f}")

  # Add current loss avg to list of losses
  if iter % plot_every == 0:
      all_losses.append(current_loss / plot_every)
      current_loss = 0

