import jieba

news_file = r'D:\datas\THUCNews\THUCNews\体育\1001.txt'


def chs_filter(values):
    ret = ''
    for uchar in values:
        if u'\u4e00' <= uchar <= u'\u9fa5':  # 是中文字符
            if uchar != ' ':  # 去除空格
                ret += uchar
    return ret


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