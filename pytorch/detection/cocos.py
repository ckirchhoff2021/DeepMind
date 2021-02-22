import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import models, transforms

train_images_path = '/Users/chenxiang/Downloads/dataset/cocos/train2014'
annotation_file = '/Users/chenxiang/Downloads/dataset/cocos/annotations/instances_train2014.json'


def analyze():
    annotations = json.load(open(annotation_file, 'r'))
    images = annotations['images']
    categories = annotations['categories']

    image_dict = dict()
    for image_info in images:
        id = image_info['id']
        file_name = image_info['file_name']
        image_dict[id] = file_name

    cate_dict = dict()
    for cat in categories:
        id = cat['id']
        supercategory = cat['supercategory']
        name = cat['name']
        cate_dict[str(id)] = {
            'supercategory': supercategory,
            'name': name
        }

    index = np.random.randint(len(images))
    annotation_info = annotations['annotations'][index]
    category = cate_dict[str(annotation_info['category_id'])]
    print(annotation_info)
    print(category)

    id = annotation_info['image_id']
    image_data = Image.open(os.path.join(train_images_path, image_dict[id]))
    draw = ImageDraw.Draw(image_data)
    x, y, w, h = annotation_info['bbox']
    v1 = [x, y]
    v2 = [x + w, y]
    v3 = [x + w, y + h]
    v4 = [x, y + h]

    draw.line([v1[0], v1[1], v2[0], v2[1]], width=5, fill=(255,0,0))
    draw.line([v2[0], v2[1], v3[0], v3[1]], width=5, fill=(255,0,0))
    draw.line([v3[0], v3[1], v4[0], v4[1]], width=5, fill=(255,0,0))
    draw.line([v4[0], v4[1], v1[0], v1[1]], width=5, fill=(255,0,0))
    image_data.show()


def main():
    pass


if __name__ == '__main__':
    # main()
    analyze()