from transformers import BertModel
import torch
import torch.nn as nn
import numpy as np

class BERTClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BERTClassifier, self).__init__()
        D_in, H, D_out = 768, 50, 2 # bert pre-train시 output_dim=768
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out))

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param      input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param      attention_mask (torch.Tensor): a tensor that hold attention mask information 
                    with shape (batch_size, max_length)
        @return     logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        """
        outputs = self.bert(input_ids, attention_mask) # (batch_size, sequence_length, hidden_size)

        last_hidden_state_cls = outputs[0][:, 0, :] # (batch_size, hidden_size)

        logits = self.classifier(last_hidden_state_cls)

        return logits

class LSTM_classifier(nn.Module):
    def __init__(self, n_layers, hidden_dim, vocab_size, embedding_dim,  output_dim, dropout, pad_idx):
        super(LSTM_classifier, self).__init__()
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim, padding_idx = pad_idx)
        
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.n_layers, dropout=dropout)
        
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, review):
        input = self.embedding(review)
        batch_size = review.shape[1]
        h = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        out, (h, c) = self.lstm(input, (h, c))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])

        return out

# Model 
class CNN_classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_kernels, kernel_sizes, output_dim, dropout, pad_idx):
        super(CNN_classifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = embedding_dim, padding_idx = pad_idx)

        self.conv1d = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim, # input channel수 ( ex RGB 이미지 = 3 )
                                              out_channels=n_kernels[ksize], # convolution에 의해 생성될 channel의 수
                                              kernel_size=kernel_sizes[i]) # ksize만 변화. embedding_dim은 고정
                                              for ksize in range(len(kernel_sizes))])

        self.fc = nn.Linear(np.sum(num_filters), output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, review):
        embedded = self.embedding(review)
        embedded = embedded.permute(0, 2, 1) # conv1d input에 맞게 input
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.conv1d]
        pooled = [F.max_pool1d(conv, conv.shape[2]) for conv in conved]

        _out = torch.cat([pool.squeeze(dim=2) for pool in pooled], dim=1)
        logits  = self.fc(self.dropout(_out))

        return logits 