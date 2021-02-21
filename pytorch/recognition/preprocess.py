import os
import json
from tqdm import tqdm
from common_path import *


def generate_datas(folder_path, folder_names, str_train, str_test):
    '''
    generate data_file.json , divide the images into training, testing
    @params:
    folder_path: image folder path
    folder_names: folder_names
    str_train: output train json file
    str_test: output test json file
    '''

    train_datas = list()
    test_datas = list()
    for index, str_label in enumerate(folder_names):
        data_folder = os.path.join(folder_path, str_label)
        image_list = os.listdir(data_folder)
        datas = list()
        for image in image_list:
            if not image.endswith('.jpg'):
                continue
            image_file = os.path.join(data_folder, image)
            datas.append([image_file, index])
        cnt = len(datas)
        mid = int(cnt * 0.1)
        train_datas.extend(datas[mid:])
        test_datas.extend(datas[:mid])

    print('train: ', len(train_datas))
    print('test: ', len(test_datas))

    with open(str_train, 'w') as f:
        json.dump(train_datas, f, ensure_ascii=False)

    with open(str_test, 'w') as f:
        json.dump(test_datas, f, ensure_ascii=False)

    print('Done....')


if __name__ == '__main__':
    folder_path = '/Users/chenxiang/Downloads/dataset/datas/sofa/corners'
    folder_names = ['left', 'right']
    str_train = os.path.join(output_path, 'sofa_train.json')
    str_test = os.path.join(output_path, 'sofa_test.json')
    generate_datas(folder_path, folder_names, str_train, str_test)

