import os
import argparse
import nltk

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torchtext.legacy import data

from transformers import BertTokenizer

from model import BERTClassifier, LSTM_classifier, CNN_classifier
from epoch import train, evaluate
from utils import set_seed, preprocessing_for_bert, tokenizer_

def main(args):

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nltk.download('punkt')
    
    # Dataset
    train_df = pd.read_csv('data/train_data.csv', sep='\t')
    val_df = pd.read_csv('data/valid_data.csv', sep='\t')
    test_df = pd.read_csv('data/test_data.csv', sep='\t')

    X_train = np.array(train_df.iloc[:, 0].tolist())
    X_val = np.array(val_df.iloc[:, 0].tolist())
    X_test = np.array(test_df.iloc[:, 0].tolist())

    y_train = np.array(train_df.iloc[:, 1].tolist()).astype(np.int64)
    y_val = np.array(val_df.iloc[:, 1].tolist()).astype(np.int64)
    y_test = np.array(test_df.iloc[:, 1].tolist()).astype(np.int64)

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)
    test_labels = torch.tensor(y_test)

    print('Tokenizing data...')
    if args.model_state == 'BERT_classifier':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        train_inputs, train_masks = preprocessing_for_bert(X_train, tokenizer)
        val_inputs, val_masks = preprocessing_for_bert(X_val, tokenizer)
        test_inputs, test_masks = preprocessing_for_bert(X_test, tokenizer)

        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_sampler = SequentialSampler(val_data)
        val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

        test_data = TensorDataset(test_inputs, test_masks, test_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.batch_size)

    else:
        REVIEW = data.Field(tokenize=tokenizer_, batch_first=True) # 배치 우선 여부(True일 경우 텐서 크기의 0번째 인덱스는 배치사이즈로 설정)
        LABEL = data.LabelField(dtype=torch.float)

        # {csv컬럼명 : (데이터 컬럼명, Field이름)} / id는 사용 x
        fields = {'REVIEW': ('REVIEW', REVIEW), 'LABEL': ('LABEL', LABEL)}

        train_data, valid_data, test_data = data.TabularDataset.splits(path = 'data', # 데이터 파일 경로 
                                                        train = './data/train_data.csv',
                                                        valid = './data/valid_data.csv',
                                                        test = './data/test_data.csv',
                                                        format = 'csv', # 데이터 파일 형식
                                                        fields = fields)

        REVIEW.build_vocab(train_data,
                        max_size = args.MAX_VOCAB_SIZE,
                        vectors = 'fasttext.simple.300d',
                        unk_init = torch.Tensor.normal_)
        LABEL.build_vocab(train_data)

        INPUT_DIM = len(REVIEW.vocab)

        train_dataloader, val_dataloader, test_dataloader = data.BucketIterator.splits(
                                                            (train_data, valid_data, test_data),
                                                            batch_size=args.batch_size,
                                                            sort_key=lambda x: len(x.REVIEW), # 길이가 유사한 것을 일괄 처리하고, 패딩을 최소화하기위해 길이로 정렬
                                                            sort_within_batch = True) # 내림차순 정렬

        PAD_IDX = REVIEW.vocab.stoi[REVIEW.pad_token]


    if args.model_state == 'BERT_classifier':
        model = BERTClassifier(freeze_bert=False).to(device)

    elif args.model_state == 'CNN_classifier':
        N_KERNELS = 100
        KERNEL_SIZES = [3, 4, 5]
        PAD_IDX = REVIEW.vocab.stoi[REVIEW.pad_token]

        model = CNN_classifier(INPUT_DIM, args.EMBEDDING_DIM, N_KERNELS, KERNEL_SIZES, args.OUTPUT_DIM, args.DROPOUT, PAD_IDX).to(device)

    elif args.model_state == 'LSTM_classifier':
        model = LSTM_classifier(args.num_layers, args.HIDDEN_DIM, INPUT_DIM, args.EMBEDDING_DIM, args.OUTPUT_DIM, args.DROPOUT, PAD_IDX).to(device)
    
    else:
        raise NotImplementedError('Model type {} is not implemented'.format(args.model_state))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    train(args, model, train_dataloader, val_dataloader, optimizer, loss_fn, evaluation=True)
    evaluate(args, model, test_dataloader, loss_fn)
    
    save_model_name = f'{args.model_state}.pt'
    torch.save(model.state_dict(), os.path.join(args.save_model_path, save_model_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_state', default='BERT_classifier', type=str,
                        help='model change')
    parser.add_argument("--save_model_path", type=str, default='./result/',
                        help="path to save model")                        
    
    parser.add_argument('--lr', default=5e-5, type=float, 
                        help='optimizer learning rate for train')
    parser.add_argument('--epochs', default=20, type=int, 
                        help='epochs for train')
    parser.add_argument('--seed', default=188, type=int,
                        help='Random seed for system')

    parser.add_argument('--batch_size', default=32, type=int, 
                        help='batch size for train')
    # parser.add_argument('--MAX_LEN', default=64, type=int, 
    #                     help='max_len of sentence')

    parser.add_argument('--num_layers', default=2, type=int, 
                        help='num_layers size for train')
    parser.add_argument('--HIDDEN_DIM', default=512, type=int, 
                        help='HIDDEN_DIM size for train')
    parser.add_argument('--EMBEDDING_DIM', default=300, type=int, 
                        help='EMBEDDING_DIM size for train')
    parser.add_argument('--DROPOUT', default=0.5, type=float, 
                        help='DROPOUT ratio')
    parser.add_argument('--OUTPUT_DIM', default=2, type=int, 
                        help='# of class')
    parser.add_argument('--MAX_VOCAB_SIZE', default=50000, type=int, 
                        help='# of MAX_VOCAB_SIZE')


    args = parser.parse_args()
    
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)

    main(args)