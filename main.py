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

batch_size = 23

def labelToTensor(label):
  return torch.tensor([mydataset.all_categories.index(label)], dtype=torch.long)

def collate_batch(batch):
  # TODO packed
  
  # print(batch) # {"name": ["sanga", "sangmin"], "label": ["Korea", "Korea"]}
  name_lengths = [len(batch[i]["name"]) for i in range(len(batch))]
  max_len = max(name_lengths)

  # packed = rnn_utils.pack_sequence(,,,)

  labels = torch.LongTensor([labelToTensor(batch[i]["label"]) for i in range(len(batch))])
  # print(labels)

  padded_names = [nameToTensor(pad(batch[i]["name"], max_len)) for i in range(len(batch))]
  names = torch.cat(padded_names, dim=1)

  return (names, labels)


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

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), dim=1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
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

n_iters = 5
print_every = 1
plot_every = 1000
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

    name_tensor, label_tensor = sample_batched
  
    hidden = model.initHidden(name_tensor.size(1))

    if useGPU:
      label_tensor = label_tensor.cuda()
      name_tensor = name_tensor.cuda()
      hidden = hidden.cuda()

    #feed into model letter by letter
    for i in range(name_tensor.size(0)):
      pred, hidden = model(name_tensor[i], hidden)

    optimizer.zero_grad()
    model.zero_grad()
    loss = criterion(pred, label_tensor)
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


  # TODO Validation
  # total_loss, total_cnt, correct_cnt = 0.0, 0.0, 0.0
  # for batch_idx, sample_batched in enumerate(valid_loader):
  #   with torch.no_grad():
  #     if useGPU:

    # output, loss = train(label_tensor, name_tensor)
    # current_loss += loss

  # Print iter number, loss, name and guess
  # if iter % print_every == 0:
  print(f"[{iter}/{n_iters}] ({timeSince(start)}) Train Loss : {train_loss_iter[iter]:.4f}")

  # Add current loss avg to list of losses
  if iter % plot_every == 0:
      all_losses.append(current_loss / plot_every)
      current_loss = 0

torch.save(model, 'char-rnn-classification.pt')