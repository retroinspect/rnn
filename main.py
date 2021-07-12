from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence

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
      label = os.path.splitext(os.path.basename(filename))[0]
      self.all_categories.append(label)
      lines = readLines(filename)

      for name in lines:
        self.data.append({'name': name, 'label': label})
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

def pad(name, max_len):
  # print(f"padding {name} with {max_len}")
  name += "0" * (max_len - len(name))
  # print(name)
  return name

mydataset = NameCountryDataset()
print(mydataset.all_categories)
# print(mydataset.name_len)

len_data = len(mydataset)

print("data size: ", len_data)

n_train_data = int(len_data * 0.9)
n_val_data = len_data - n_train_data
train_dataset, val_dataset = random_split(mydataset, [n_train_data, n_val_data])

batch_size = 8

def labelToTensor(label):
  return torch.tensor([mydataset.all_categories.index(label)], dtype=torch.long)

def collate_batch(batch):
  # print(batch) # {"name": ["sanga", "sangmin"], "label": ["Korea", "Korea"]}

  names = [ batch[i]["name"] for i in range(len(batch)) ]
  name_lengths = torch.LongTensor([len(name) for name in names])
  max_len = max(name_lengths)
  names = torch.cat([nameToTensor(pad(name, max_len)) for name in names], dim=1)
  
  labels = torch.LongTensor([labelToTensor(batch[i]["label"]) for i in range(len(batch))])

  name_lengths, sorted_idx = name_lengths.sort(0, descending=True)
  names = names[:, sorted_idx]
  labels = labels[sorted_idx]

  # TODO update_bounds 좀 더 효율적으로 짜기
  update_bounds = torch.LongTensor([0] * max_len) # 한 번에 process하는 글자 개수 
  for name_len in name_lengths:
    tmp = torch.LongTensor([1] * name_len + [0] * (max_len - name_len))
    update_bounds += tmp;

  return (names, labels, update_bounds)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

def letterToIndex(letter):
  if (letter == '0'):
    return 0
  return all_letters.find(letter) + 1

# one-hot vector
def nameToTensor(line):
  tensor = torch.zeros(len(line), 1, n_letters)
  for li, letter in enumerate(line):
    tensor[li][0][letterToIndex(letter)] = 1
  return tensor

def thresholding(prediction):
  confidence, pred_label = torch.max(prediction, 1)
  return pred_label

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, update_bound=None):
        old_hidden = hidden
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)

        # update_bound 미만의 index에 대해서만 hidden update
        # 이상의 index에 대해서는 old_hidden 사용
        if (update_bound != None):
          hidden = torch.cat([hidden[0:update_bound, :], old_hidden[update_bound:, :]])
        # update_bound = 4 이면 hidden[3, :] + old_hidden[4:, :] 

        return output, hidden

    def initHidden(self, sample_batched_size):
      return torch.zeros(sample_batched_size, self.hidden_size)

n_hidden = 128
n_categories = 18
model = RNN(n_letters, n_hidden, n_categories)

if useGPU:
  model.to(device)

def labelFromOutput(output, dataset):
  top_n, top_i = output_topk(1)
  label_i = top_i[0].item()
  return dataset.all_categories[label_i], label_i

criterion = nn.NLLLoss()
learning_rate = 0.005 # for SGD
# learning_rate = 0.0001 # for Adam

import time
import math

n_iters = 20
print_every = 1
plot_every = 1000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()
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

    name_tensor, label_tensor, update_bounds = sample_batched
    hidden = model.initHidden(name_tensor.size(1))

    if useGPU:
      label_tensor = label_tensor.cuda()
      name_tensor = name_tensor.cuda()
      update_bounds = update_bounds.cuda()
      hidden = hidden.cuda()

    # feed into model letter by letter
    for i in range(name_tensor.size(0)):
      pred, hidden = model(name_tensor[i], hidden, update_bounds[i])
      # pred, hidden = model(name_tensor[i], hidden, None) # 비교용


    optimizer.zero_grad()
    model.zero_grad()
    loss = criterion(pred, label_tensor)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() # accumulate loss
    total_cnt += label_tensor.size(0) # accumulate the number of data
    correct_cnt += (label_tensor == thresholding(pred)).sum().item() # number of correct

  accuracy = correct_cnt * 1.0 / total_cnt  # calculate accuracy  (#accumulated-correct-prediction/#accumulated-data)
  train_loss_iter[iter] = total_loss / total_cnt # calculate and save loss (#accumulated-loss/#accumulated-data)
  train_accuracy_iter[iter] = accuracy  # save accuracy

  # Validation
  total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0
  for batch_idx, sample_batched in enumerate(valid_loader):
    with torch.no_grad():
      name_tensor, label_tensor, update_bounds = sample_batched  
      hidden = model.initHidden(name_tensor.size(1))
      if useGPU:
        label_tensor = label_tensor.cuda()
        name_tensor = name_tensor.cuda()
        update_bounds = update_bounds.cuda()
        hidden = hidden.cuda()

      for i in range(name_tensor.size(0)):
        pred, hidden = model(name_tensor[i], hidden, update_bounds[i])
        # pred, hidden = model(name_tensor[i], hidden, None) # 비교용

    loss = criterion(pred, label_tensor)
    total_loss += loss.item() # accumulate loss
    total_cnt += label_tensor.size(0) # accumulate the number of data
    correct_cnt += (label_tensor == thresholding(pred)).sum().item() # number of correct
      
  accuracy = correct_cnt * 1.0 / total_cnt  # calculate accuracy  (#accumulated-correct-prediction/#accumulated-data)
  valid_loss_iter[iter] = total_loss / total_cnt # calculate and save loss (#accumulated-loss/#accumulated-data)
  valid_accuracy_iter[iter] = accuracy  # save accuracy

  # Print iter number, loss
  print(f"[{iter}/{n_iters}] Train Loss : {train_loss_iter[iter]:.4f} Train Acc : {train_accuracy_iter[iter]:.2f} \
  Valid Loss : {valid_loss_iter[iter]:.4f} Valid Acc : {valid_accuracy_iter[iter]:.2f}")


torch.save(model, 'char-rnn-classification.pt')