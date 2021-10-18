import json
import os
import numpy as np
from tqdm import tqdm

import jieba
import pickle
from tqdm import tqdm
from py2neo import Graph, Node, Relationship, NodeMatcher, RelationshipMatcher


def main():
    # g = Graph('http://11.165.235.244:7474', username="neo4j", password="cro_kg")
    g = Graph('http://11.165.235.244:7474', auth=("neo4j", "cro_kg"))

    # matcher_n = NodeMatcher(g)
    # matcher_n.match("宗教")
    print(g)

    values = g.run("MATCH (n1)-[r]->(n2) RETURN r.name, n1.name, n2.name")
    # values = g.run("MATCH (n) RETURN n")
    # values = g.run("MATCH (n1)-[r]->(n2) RETURN r")
    # print(type(values))
    # data = values.data()
    # print(len(data))


    with open('output/kg.json', 'w') as f:
        triples = list()
        for i in tqdm(values):
            r = str(i[0])
            n1 = str(i[1])
            n2 = str(i[2])

            triples.append({
                'x1': n1,
                'r': r,
                'y1': n2
            })
        json.dump(triples, f, ensure_ascii=False)


def get_last_no_empty(arr):
    count = len(arr)
    i = count -1
    while i >= 0:
        v = arr[i]
        if len(v) > 0:
            return v
        i = i-1
    return ''


def process():
    kg_arr = json.load(open('output/kg.json', 'r'))
    print('==> Num: ', len(kg_arr))

    node_list = set()
    triples = list()
    for value in kg_arr:
        x1 = value['x1'].replace(' ', '')
        r = value['r'].replace(' ', '')
        y1 = value['y1'].replace(' ', '')

        if "文革" not in x1 and "文革" not in y1:
            continue

        if x1 in ['', 'none', 'None'] or y1 in ['', 'none', 'None'] or r in ['None', 'none', '']:
            continue

        # if "战争" in x1 or "战争" in y1:
        v1 = x1
        if '_' in x1:
            v1s = x1.split('_')
            v1 = get_last_no_empty(v1s)

        v2 = y1
        if '_' in y1:
            v2s = y1.split('_')
            v2 = get_last_no_empty(v2s)

        c = r
        if '_' in r:
            cs = r.split('_')
            c = get_last_no_empty(cs)


        if v1 == '' or v2 == '' or c == '':
            continue

        node_list.add(v1)
        node_list.add(v2)

    # with open('output/kg_v2.json', 'w') as f:
    #     json.dump(triples, f, ensure_ascii=False)

    # print('done.....')

    print(node_list)


def cut(value):
    slices = jieba.cut(value, cut_all=False)
    arr = set(slices)
    return arr


def split():
    u = "得，拼,廊街有售,白云普月,冰月的里具,x人家哪个游戏没几个脚本,全部,好看视频,小树苗,占地格数,收获次数,适宜季节"
    test1 = jieba.cut(u, cut_all=True)
    print("全模式: " + "| ".join(test1))

    test2 = jieba.cut(u, cut_all=False)
    print("精确模式: " + "| ".join(test2))

    test3 = jieba.cut_for_search(u)
    print("搜索引擎模式:" + "| ".join(test3))

    values = cut(u)
    print(values)


def count():
    datas = json.load(open('output/kg_v2.json', 'r'))
    print(len(datas))


def cal_tf_idf():
    pass


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def text_split():
    root_path = '/Users/chenxiang/Downloads/dataset/youku'
    text_file = os.path.join(root_path, 'youku_clip_text.pkl')
    text_datas = pickle.load(open(text_file, 'rb'))
    print('==> Num: ', len(text_datas.keys()))

    sample_file = os.path.join(root_path, 'youku_clip_black_sample.json')
    blacks = json.load(open(sample_file, 'r'))
    output_dict = dict()
    for ikey in blacks.keys():
        videos = blacks[ikey]
        risk_dict = dict()
        for vid in tqdm(videos):
            text = text_datas[vid]
            try:
                values = jieba.cut(text, cut_all=False)
                for vt in values:
                    if vt is None or vt in [',', ',', ' '] or len(vt) == 0:
                        continue

                    if not is_all_chinese(vt) or len(vt) == 1:
                        continue

                    if vt not in risk_dict.keys():
                        risk_dict[vt] = 0

                    risk_dict[vt] += 1
            except:
                print('==> Error happened, continue ...  ')

        output_dict[ikey] = risk_dict

    with open(os.path.join(root_path, 'youku_jieba_sample_v2.json'), 'w') as f:
        json.dump(output_dict, f, ensure_ascii=False)


def preprocess():
    root_path = '/Users/chenxiang/Downloads/dataset/youku'
    train_datas = pickle.load(open(os.path.join(root_path, 'youku_clip_label_train.pkl'), 'rb'))
    print('==> Train Num: ', len(train_datas))

    val_datas = pickle.load(open(os.path.join(root_path, 'youku_clip_label_val.pkl'), 'rb'))
    print('==> Test Num: ', len(val_datas))

    print(' ------  ')
    sample_dict = dict()
    for ikey in train_datas.keys():
        label = train_datas[ikey]
        if label == 0:
            continue
        k = str(label)
        if k not in sample_dict.keys():
            sample_dict[k] = []
        sample_dict[k].append(ikey)

    for ikey in val_datas.keys():
        label = val_datas[ikey]
        if label == 0:
            continue
        k = str(label)
        if k not in sample_dict.keys():
            sample_dict[k] = []
        sample_dict[k].append(ikey)


    count = 0
    values = list()
    for k in sample_dict.keys():
        count += len(sample_dict[k])
        values.append((k, len(sample_dict[k])))


    arrs = sorted(values, key=lambda x: x[1], reverse=True)
    for v in arrs:
        print(v[0], v[1])

    # with open('/Users/chenxiang/Downloads/dataset/youku/youku_clip_black_sample.json', 'w') as f:
    #     json.dump(sample_dict, f, ensure_ascii=False)

    print('Done ...')



def printf(index):
    sample_dict = json.load(open('/Users/chenxiang/Downloads/dataset/youku/youku_jieba_sample_v2.json', 'r'))
    key = str(index)
    values = list(sample_dict[key].items())
    output = sorted(values, key=lambda x:x[1], reverse=True)
    print(output)



def filter(values, vocab_dict):
    for word in values:
        if word is None or word in [',', '.', ''] or len(word) == 0:
            continue

        if not is_all_chinese(word) or len(word) == 1:
            continue

        if word not in vocab_dict.keys():
            vocab_dict[word] = 0

        vocab_dict[word] += 1


def words_cut(file, save_file):
    texts_dict = pickle.load(open(file, 'rb'))
    print('==> Num: ', len(texts_dict))

    vocab_dict = dict()

    for ikey in texts_dict.keys():
        ivalue = dict()
        ocr, asr = texts_dict[ikey]

        if len(ocr) > 0:
            ocr_values = jieba.cut(ocr, cut_all=False)
            filter(ocr_values, ivalue)

        if len(asr) > 0:
            asr_values = jieba.cut(asr, cut_all=False)
            filter(asr_values, ivalue)

        vocab_dict[ikey] = ivalue

    with open(save_file, 'w') as f:
        json.dump(vocab_dict, f, ensure_ascii=False)

    print('Done ... ')


def cut_test():
    import time
    words = '从前有一座山，我从山门前经过，出现一个和尚'
    t1 = time.time()
    word_list = jieba.cut(words, cut_all=False)
    t2 = time.time()

    for w in word_list:
        print(w)

    print('cost: ', t2 - t1)



if __name__ == '__main__':
    # main()
    # process()
    # count()
    # text_split()
    # split()
    # preprocess()
    # count()

    # print(is_all_chinese("龍"))
    # printf(17)

    cut_test()
