import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import torch

from torch.utils.data import Dataset

# class CustomDataset(Dataset):
#     def __init__(self, args, ):
#         self.X = pd.read_csv('f{args.data_path}.csv', sep='\t')
#         self.y = []     
                
        

#     def __len__(self):
#         return len(self.X)

#     def __getitem__(self, idx):
#         return self.X[idx], self.y[idx]

train_df = pd.read_csv('data/train_data.csv', sep='\t')
val_df = pd.read_csv('data/valid_data.csv', sep='\t')
test_df = pd.read_csv('data/test_data.csv', sep='\t')

X_train = np.array(train_df.iloc[:, 0].tolist())
X_val = np.array(val_df.iloc[:, 0].tolist())
X_test = np.array(test_df.iloc[:, 0].tolist())

y_train = np.array(train_df.iloc[:, 1].tolist()).astype(np.int64)
y_val = np.array(val_df.iloc[:, 1].tolist()).astype(np.int64)
y_test = np.array(test_df.iloc[:, 1].tolist()).astype(np.int64)

# Print sentence 0 and its encoded token ids
# token_ids = list(preprocessing_for_bert([X_train[0]])[0].squeeze().numpy())
# print('Original: ', X_train[0])
# print('Token IDs: ', token_ids)


