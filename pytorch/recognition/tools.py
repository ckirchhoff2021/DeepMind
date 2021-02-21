import json
import random
import requests
from PIL import Image
from multiprocessing import Pool, Process


def download_image(str_image, url, folder):
    '''
    downlong image and resize to 256 x 256
    '''
    image_url = url
    try:
        src = requests.get(image_url)
        s1 = Image.open(BytesIO(src.content))
        s2 = s1.resize((256, 256))
        dst = os.path.join(folder,  str_image)
        s1.save(dst)
    except:
        print('Error:', str_image, url)


def convert_RGBA2RGB(src):
    '''
    convert rgba image to rgb
    '''
    s1 = Image.open(src)
    s2 = s1.resize((256, 256))
    alpha = Image.new('RGBA', s2.size, (255, 255, 255))
    s3 = Image.alpha_composite(alpha, s2)
    s3 = s3.convert('RGB')
    return s3


