import json
import os
import glob
import jieba
from tqdm import *
from multiprocessing import Process

news_file = r'D:\datas\THUCNews\THUCNews\体育\1001.txt'

txt_labels = ["体育", "娱乐" ,"家居" , "彩票" , "房产", "教育" ,"时尚", "时政" ,"星座" ,"游戏" ,"社会" ,"科技","股票","财经"]
invalid_words = ["了", "你", "我", "他", "的", "年", "月", "日", "可能", "会", "我们", "他们", "你们", "这", "那", "时",
                 "请", "让", "而且", "和", "而"]


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
    x = "简直了，现在很不开心"
    words = list(jieba.cut(x))
    print(words)
    with open(news_file, 'r', encoding='utf-8') as f:
        data = f.read()
        print(data)
        data = chs_filter(data)
        print(data)
        words = list(jieba.cut(data))
        print(words)


if __name__ == '__main__':
    word_segmentation()
