import os

import torch
import torch.nn as nn

from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW, BertConfig


PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-base-multilingual': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz"
}

PRETRAINED_VOCAB_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
    'bert-base-multilingual': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-vocab.txt",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt"
}




class BertNN(nn.Module):
    def __init__(self, classes=10):
        super(BertNN, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.bert = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(self.config.hidden_size, classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        out_pool = outputs[1]
        logit = self.fc(out_pool)
        return logit


def main():
    bert_path = '/Users/chenxiang/Downloads/bert/chinese'
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    print('bert testing...')
    encode_dict = tokenizer.encode_plus('有一天', '你变成了猪')
    print(encode_dict)
    print(tokenizer.convert_ids_to_tokens(encode_dict['input_ids']))

    token_tensor = torch.tensor([encode_dict['input_ids']])
    seg_tensor = torch.tensor(encode_dict['token_type_ids'])
    print(token_tensor.size())
    print(seg_tensor.size())

    config = BertConfig.from_pretrained(bert_path)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel.from_pretrained(bert_path, config=config)

    outputs = model(token_tensor, token_type_ids=seg_tensor)
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2][0].shape)
    print(outputs[3][0].shape)


def test():
    root = '/Users/chenxiang/Downloads/bert/chinese/'
    tokenizer = BertTokenizer.from_pretrained(root)
    config = BertConfig.from_pretrained(root)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel.from_pretrained(root, config=config)

    print(tokenizer.encode('陛下驾到'))
    sen_code = tokenizer.encode_plus('这个故事没有终点', '正如星空没有彼岸')
    print(sen_code)
    print(tokenizer.convert_ids_to_tokens(sen_code['input_ids']))

    token_tensor = torch.tensor([sen_code['input_ids']])
    segments_tensors = torch.tensor(sen_code['token_type_ids'])
    print(token_tensor.size())
    print(segments_tensors.size())

    model.eval()
    with torch.no_grad():
        outputs = model(token_tensor, token_type_ids=segments_tensors)
        encode_layers = outputs
        print(type(encode_layers))
        print(encode_layers[0].shape)
        print(encode_layers[1].shape)
        print(encode_layers[2][0].shape)
        print(encode_layers[3][0].shape)



if __name__ == '__main__':
    main()
    # test()