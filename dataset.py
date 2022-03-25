import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import torch

from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# IMDB dataset
total_data = pd.read_csv('IMDB Dataset.csv', engine='python', error_bad_lines=False)

le = LabelEncoder()

y = le.fit_transform(total_data['sentiment'])


total_data['sentiment'] = y

# csv 파일로 저장
X_train, X_test, y_train, y_test = train_test_split(total_data['review'], total_data['sentiment'],test_size=0.3,
                                                    stratify=total_data['sentiment'], random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3,
                                                    stratify=y_train, random_state=0)

train_data = pd.concat([X_train, y_train], axis=1)
valid_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data1/train_data.csv', sep='\t', index=False)
valid_data.to_csv('data1/valid_data.csv', sep='\t', index=False)
test_data.to_csv('data1/test_data.csv', sep='\t', index=False)

# convervative & liberal dataset
conservative_data = pd.read_csv('raw_data/conservative.txt', sep='\t', header=None).dropna()
liberal_data = pd.read_csv('raw_data/liberal.txt', sep='\t', header=None).dropna()

# column 이름 변경 
conservative_data.rename(columns = {0: 'LABEL', 1 : 'REVIEW'}, inplace = True)
liberal_data.rename(columns = {0: 'LABEL', 1 : 'REVIEW'}, inplace = True)
liberal_data['LABEL'] = 1 # label 구분

# liberal_data.head()
total_data = pd.concat([conservative_data, liberal_data])

# csv 파일로 저장
X_train, X_test, y_train, y_test = train_test_split(total_data['REVIEW'], total_data['LABEL'],test_size=0.3,
                                                    stratify=total_data['LABEL'], random_state=0)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3,
                                                    stratify=y_train, random_state=0)

train_data = pd.concat([X_train, y_train], axis=1)
valid_data = pd.concat([X_val, y_val], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

train_data.to_csv('data/train_data.csv', sep='\t', index=False)
valid_data.to_csv('data/valid_data.csv', sep='\t', index=False)
test_data.to_csv('data/test_data.csv', sep='\t', index=False)