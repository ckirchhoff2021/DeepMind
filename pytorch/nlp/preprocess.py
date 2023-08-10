import json
import os
import glob
import random
import jieba
from tqdm import *
from multiprocessing import Process
from transformers import BertTokenizer,BertModel, BasicTokenizer, BertForMaskedLM

txt_labels = ["体育", "娱乐" ,"家居" , "彩票" , "房产", "教育" ,"时尚", "时政" ,"星座" ,"游戏" ,"社会" ,"科技","股票","财经"]
invalid_words = ["了", "你", "我", "他", "的", "年", "月", "日", "可能", "会", "我们", "他们", "你们", "这", "那", "时",
                 "请", "让", "而且", "和", "而"]
data_folder = r'/home/cx/A100/datas/THUCNews/'
out_folder = r"/home/cx/A100/datas/words"

def chs_filter(values):
    ret = ''
    for uchar in values:
        if u'\u4e00' <= uchar <= u'\u9fa5':  # 是中文字符
            if uchar != ' ':  # 去除空格
                ret += uchar
    return ret

def words_filter(words):
    ret = list()
    for x in words:
        if x in invalid_words:
            continue
        ret.append(x)
    return ret


def multiprocess_cut():
    def handle(label, i):
        vocabulary = dict()
        folder = os.path.join(data_folder, label)
        files = list(glob.glob(os.path.join(folder, '*.txt')))
        print('==> ', label, len(files))
        cut_folder = os.path.join(out_folder, str(i))
        if not os.path.exists(cut_folder):
            os.mkdir(cut_folder)
        for j, file in tqdm(enumerate(files)):
            with open(file, 'r') as fr:
                sentences = fr.read()
                words = chs_filter(sentences)
                segments = jieba.lcut(words, cut_all=False, HMM=True)
                remain = words_filter(segments)
                with open(os.path.join(cut_folder, f'{j}.json'), 'w') as fw:
                    json.dump(remain, fw, ensure_ascii=False)
                for x in remain:
                    frequency = vocabulary.get(x, 0) + 1
                    vocabulary[x] = frequency
        with open(os.path.join(cut_folder, 'vocab.json'), 'w') as f:
            json.dump(vocabulary, f, ensure_ascii=False)

    for i, label in enumerate(txt_labels):
        p = Process(target=handle, args=(label, i))
        p.start()
    print('Done ......')


def word_segmentation():
    vocabulary = dict()
    for i, label in enumerate(txt_labels):
        folder = os.path.join(data_folder, label)
        files = list(glob.glob(os.path.join(folder, '*.txt')))
        print('==> ', label, len(files))
        cut_folder = os.path.join(out_folder, str(i))
        if not os.path.exists(cut_folder):
            os.mkdir(cut_folder)
        for j, file in tqdm(enumerate(files)):
            with open(file ,'r') as fr:
                sentences = fr.read()
                words = chs_filter(sentences)
                segments = jieba.lcut(words, cut_all=False, HMM=True)
                remain = words_filter(segments)
                with open(os.path.join(cut_folder, f'{j}.json'), 'w') as fw:
                    json.dump(remain, fw, ensure_ascii=False)
                for x in remain:
                    frequency = vocabulary.get(x, 0) + 1
                    vocabulary[x] = frequency
    with open(os.path.join(out_folder, 'vocab.json'), 'w') as f:
        json.dump(vocabulary, f, ensure_ascii=False)
    print('Done ......')


def merge_vocabulary():
    vocabulary = dict()
    for i in range(14):
        vocab = os.path.join(out_folder, str(i), 'vocab.json')
        with open(vocab, 'r') as f:
            vocabs = json.load(f)
            for ikey in vocabs:
                if ikey not in vocabulary:
                    vocabulary[ikey] = 0
                vocabulary[ikey] += vocabs[ikey]
    vocab_list = list(vocabulary.items())
    vocab_list = sorted(vocab_list, key=lambda x: x[1])
    print(vocab_list)
    word = [v[0] for v in vocab_list]
    count = [v[1] for v in vocab_list]
    save_vocab = {
        'word': word,
        'frequency': count
    }
    with open(os.path.join(out_folder, 'vocab.json'), 'w') as f:
        json.dump(save_vocab, f, ensure_ascii=False)
    print('Done ...')


def glob_files():
    datas = dict()
    for i, label in enumerate(txt_labels):
        folder = os.path.join(data_folder, label)
        files = list(glob.glob(os.path.join(folder, '*.txt')))
        print('==> ', label, len(files))
        datas[str(i)] = files
    with open(os.path.join(out_folder, 'datas.json'), 'w') as f:
        json.dump(datas, f, ensure_ascii=False)
    print('Done ...')


def pad_sequence(sequence, max_length=512):
    out_sequence = None
    mask = None
    if len(sequence) >= max_length:
        out_sequence = sequence[:max_length]
        mask = [1] * max_length
    else:
        out_sequence = sequence + (max_length - len(sequence)) * [0]
        mask = [1] * len(sequence) + [0] * (max_length - len(sequence))
    return out_sequence, mask


def gen_dataset():
    datas = json.load(open(os.path.join(out_folder, 'datas.json'), 'r'))
    tokenizer = BertTokenizer.from_pretrained("bert-chinese")
    def tokenize(files, label):
        token_ids = list()
        token_masks = list()
        for file in files:
            with open(file, 'r') as f:
                sequence = tokenizer.encode(f.read())
                ids, mask = pad_sequence(sequence)
                token_ids.append(ids)
                token_masks.append(mask)
        result = {
            'id': token_ids,
            'mask': token_masks
        }
        with open(os.path.join(out_folder, f'{label}.json'), 'w') as fw:
            json.dump(result, fw, ensure_ascii=False)

    for i in range(14):
        print(i, len(datas[str(i)]))
        p = Process(target=tokenize, args=(datas[str(i)], str(i)))
        p.start()
    print('done ...')


def generate():
    train_dict = dict()
    val_dict = dict()
    for i in range(14):
        file = os.path.join(out_folder, f"{i}.json")
        data = json.load(open(file, 'r'))
        indices = data['id']
        random.shuffle(indices)
        num = len(indices)
        print('Num: ', i, len(indices))
        pos = int(0.9 * num)
        train_list = indices[:pos]
        val_list = indices[pos:]
        train_dict[str(i)] = train_list
        val_dict[str(i)] = val_list
    with open(os.path.join(out_folder, 'train.json'), 'w') as f:
        json.dump(train_dict, f, ensure_ascii=False)
    with open(os.path.join(out_folder, 'val.json'), 'w') as f:
        json.dump(val_dict, f, ensure_ascii=False)


def bert_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-chinese")
    file = r'/home/cx/A100/datas/THUCNews/体育/112469.txt'
    with open(file, 'r') as f:
        sentences = f.read()
        print('sentences length: ', len(sentences))
        tokens = tokenizer.tokenize(sentences)
        print('tokens ...')
        print('tokens length: ', len(tokens))
        inputs = tokenizer.encode(sentences, add_special_tokens=True)
        print(inputs)

        # inputs = tokenizer.encode_plus(sentences, add_special_tokens=True)
        # print('encodes ... ')
        # print(len(inputs['input_ids']))
        # print(inputs['input_ids'])

        # print(len(inputs['token_type_ids']))
        # print(inputs['token_type_ids'])

        # print(len(inputs['attention_mask']))
        # print(inputs['attention_mask'])


if __name__ == '__main__':
    # word_segmentation()
    # multiprocess_cut()
    # merge_vocabulary()
    # bert_tokenizer()
    # glob_files()
    # gen_dataset()
    generate()
