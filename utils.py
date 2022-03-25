import random
import numpy as np
import nltk
from nltk import word_tokenize
import torch
# from transformers import get_linear_schedule_with_warmup

# Run function `preprocessing_for_bert` on the train set and the validation set
def preprocessing_for_bert(data, tokenizer):
    """Perform required preprocessing steps for pretrained BERT.
    @param      data (np.array): Array of texts to be processed.
    @return     input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return     attention_masks (torch.Tensor): Tensor of indices specifying which
                tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    MAX_LEN = 64

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs

        encoded_sent = tokenizer.encode_plus(text=sent,  # Preprocess sentence
                                            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                                            max_length=MAX_LEN,                  # Max length to truncate/pad
                                            pad_to_max_length=True,         # Pad sentence to max length
                                            #return_tensors='pt',           # Return PyTorch tensor
                                            return_attention_mask=True)      # Return attention mask

        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def tokenizer_(args, text):
    if args.model_state == 'CNN_classifier':
        token = word_tokenize(text)
        if len(token) < max(args.KERNEL_SIZE): 
            for i in range(0, max(args.KERNEL_SIZE)-len(token)):
                token.append('<PAD>') # 커널 사이즈 보다 문장의 길이가 작은 경우 에러 방지
    else:
        token = word_tokenize(text)
    return token

def set_seed(seed_value=188):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)