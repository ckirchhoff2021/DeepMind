import os
import json
import torch
import torch.nn as nn
import numpy as np

from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertForMaskedLM, AdamW, BertConfig, BertForNextSentencePrediction, BertForQuestionAnswering


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
    config = BertConfig.from_pretrained(bert_path)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel.from_pretrained(bert_path, config=config)
    model.eval()

    print('bert testing...')
    i1 = '我早上吃了饼'
    i2 = '本拉登是坏人'

    e1 = tokenizer.encode(i1)
    e2 = tokenizer.encode(i2)

    t1 = torch.tensor([e1])
    t2 = torch.tensor([e2])

    y1 = model(t1)
    y2 = model(t2)

    y1 = y1[1]
    y2 = y2[1]

    y1 = y1 / y1.norm(dim=1)
    y2 = y2 / y2.norm(dim=1)

    score = torch.mm(y1, y2.t())
    print(score)




'''
sequence_output : 输出序列,torch.Size([1, 19, 768])
pooled_output   : 对输出序列进行pool操作, torch.Size([1, 768])
hidden_states   : 隐藏层状态tuple， 13 * torch.Size([1, 19, 768]) 
attentions      : tuple, 12 * torch.Size([1, 12, 19, 19])
'''

def test():
    root = '/Users/chenxiang/Downloads/bert/chinese/'
    tokenizer = BertTokenizer.from_pretrained(root)
    config = BertConfig.from_pretrained(root)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel.from_pretrained(root, config=config)

    tokens = tokenizer.tokenize('陛下驾到,哈哈哈')
    ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]'])
    print(ids)
    print(tokenizer.encode('陛下驾到,哈哈哈'))
    print(tokenizer.convert_ids_to_tokens(tokenizer.encode('陛下驾到,哈哈哈')))
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



def test2():
    '''
    Masked Language Model
    '''
    model_name = '/Users/chenxiang/Downloads/bert/chinese/'  # 指定需下载的预训练模型参数

    # 任务一：遮蔽语言模型
    # BERT 在预训练中引入 [CLS] 和 [SEP] 标记句子的 开头和结尾
    samples = ['[CLS] 中国的首都是哪里？ [SEP] 北京是 [MASK] 国的首都。 [SEP]']  # 准备输入模型的语句

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer_text = [tokenizer.tokenize(i) for i in samples]  # 将句子分割成一个个token，即一个个汉字和分隔符
    # [['[CLS]', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '[SEP]', '北', '京', '是', '[MASK]', '国', '的', '首', '都', '。', '[SEP]']]
    # print(tokenizer_text)

    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenizer_text]
    input_ids = torch.LongTensor(input_ids)
    # print(input_ids)
    # tensor([[ 101,  704, 1744, 4638, 7674, 6963, 3221, 1525, 7027, 8043,  102, 1266,
    #           776, 3221,  103, 1744, 4638, 7674, 6963,  511,  102]])

    # 读取预训练模型
    model = BertForMaskedLM.from_pretrained(model_name, cache_dir='/Users/chenxiang/Downloads/bert')
    model.eval()

    outputs = model(input_ids)
    prediction_scores = outputs[0]  # prediction_scores.shape=torch.Size([1, 21, 21128])
    sample = prediction_scores[0].detach().numpy()  # (21, 21128)

    # 21为序列长度，pred代表每个位置最大概率的字符索引
    pred = np.argmax(sample, axis=1)  # (21,)
    # ['，', '中', '国', '的', '首', '都', '是', '哪', '里', '？', '。', '北', '京', '是', '中', '国', '的', '首', '都', '。', '。']
    print(tokenizer.convert_ids_to_tokens(pred))
    print(tokenizer.convert_ids_to_tokens(pred)[14])  # 被标记的[MASK]是第14个位置, 中


'''
问答
'''
def test3():
    # sen_code1 = tokenizer.encode_plus('今天天气怎么样？', '今天天气很好！')
    # sen_code2 = tokenizer.encode_plus('明明是我先来的！', '我喜欢吃西瓜！')

    # tokens_tensor = torch.tensor([sen_code1['input_ids'], sen_code2['input_ids']])
    # print(tokens_tensor)
    # tensor([[ 101,  791, 1921, 1921, 3698, 2582,  720, 3416,  102,  791, 1921, 1921,
    #          3698, 2523, 1962,  102],
    #         [ 101, 3209, 3209, 3221, 2769, 1044, 3341, 4638,  102, 7471, 3449,  679,
    #          1963, 1921, 7360,  102]])

    # 上面可以换成
    model_name = '/Users/chenxiang/Downloads/bert/chinese/'
    samples = ["[CLS]天气真的好啊！[SEP]一起出去玩吧！[SEP]", "[CLS]小明今年几岁了[SEP]我不喜欢学习！[SEP]"]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenized_text = [tokenizer.tokenize(i) for i in samples]
    input_ids = [tokenizer.convert_tokens_to_ids(i) for i in tokenized_text]
    tokens_tensor = torch.LongTensor(input_ids)

    # 读取预训练模型
    model = BertForNextSentencePrediction.from_pretrained(model_name, cache_dir='/Users/chenxiang/Downloads/bert')
    model.eval()

    outputs = model(tokens_tensor)
    # sequence_output：输出序列
    seq_relationship_scores = outputs[0]  # seq_relationship_scores.shape: torch.Size([2, 2])
    sample = seq_relationship_scores.detach().numpy()  # sample.shape: [2, 2]

    pred = np.argmax(sample, axis=1)
    print(pred)  # [0 0]， 0表示是上下句关系，1表示不是上下句关系


'''
句子预测
'''
def test4():
    model_name = '/Users/chenxiang/Downloads/bert/chinese/'

    # 通过词典导入分词器
    tokenizer = BertTokenizer.from_pretrained(model_name)
    # 导入配置文件
    model_config = BertConfig.from_pretrained(model_name)
    # 最终有两个输出，初始位置和结束位置
    model_config.num_labels = 2

    # 根据bert的 model_config 新建 BertForQuestionAnsering
    model = BertForQuestionAnswering(model_config)
    model.eval()

    question, text = '里昂是谁？', '里昂是一个杀手。'

    sen_code = tokenizer.encode_plus(question, text)

    tokens_tensor = torch.tensor([sen_code['input_ids']])
    segments_tensors = torch.tensor([sen_code['token_type_ids']])  # 区分两个句子的编码（上句全为0，下句全为1）

    start_pos, end_pos = model(tokens_tensor, token_type_ids=segments_tensors)
    # 进行逆编码，得到原始的token
    all_tokens = tokenizer.convert_ids_to_tokens(sen_code['input_ids'])
    print(all_tokens)  # ['[CLS]', '里', '昂', '是', '谁', '[SEP]', '里', '昂', '是', '一', '个', '杀', '手', '[SEP]']

    # 对输出的答案进行解码的过程
    answer = ' '.join(all_tokens[torch.argmax(start_pos): torch.argmax(end_pos) + 1])

    # 每次执行的结果不一致，这里因为没有经过微调，所以效果不是很好，输出结果不佳，下面的输出是其中的一种。
    print(answer)  # 一 个 杀 手



def test5():
    root = '/Users/chenxiang/Downloads/bert/base/'
    tokenizer = BertTokenizer.from_pretrained(root)
    config = BertConfig.from_pretrained(root)
    config.output_hidden_states = True
    config.output_attentions = True
    model = BertModel.from_pretrained(root, config=config)
    model.eval()

    flickr_file = '../mm/flickr30_annotations.json'
    flickr_dict = json.load(open(flickr_file, 'r'))
    tokens = flickr_dict['1001773457.jpg']
    print(tokens)

    tensor_list = list()
    for i in range(5):
        encoder = tokenizer.encode(tokens[i])
        token_tensor = torch.tensor([encoder])
        outputs = model(token_tensor)
        tensor_list.append(outputs[1])

    A = tensor_list[0]
    B = tensor_list[3].t()
    A = A / torch.norm(A, dim=1)
    B = B / torch.norm(B, dim=0)

    score = torch.mm(A, B)
    print('Similar:', score)

    token = 'you are an idiot .'
    encoder = tokenizer.encode(token)
    print(encoder)
    x = torch.tensor([encoder])
    y = model(x)
    C = y[1].t() / torch.norm(y[1].t(), dim=0)
    score = torch.mm(A, C)
    print('Random: ', score)




if __name__ == '__main__':
    # main()
    test()
    # test2()
    # test3()
    # test5()