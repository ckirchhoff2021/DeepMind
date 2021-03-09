import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

class Rtts(object):
    data_format = 'VOC'
    voc_root = 'datasets/RTTS'
    train_split = 'train.txt'
    val_split = 'all.txt' 
    class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    img_format = 'png'

    width = 480
    height = 480

    train_transform = A.Compose(
        [
            A.Resize(height=1024, width=1024, p=1),  # 896
            A.RandomSizedCrop(min_max_height=(600, 800), height=1024, width=1024, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2, 
                                        val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, 
                                            contrast_limit=0.2, p=0.9),
            ],p=0.9),
            A.ToGray(p=0.01),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(height=height, width=width, p=1),
            # A.Cutout(num_holes=5, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        ),
    )

    val_transform = A.Compose(
        [
            A.Resize(height=height, width=width, p=1.0),
            # A.LongestMaxSize(416, p=1.0),
            # A.PadIfNeeded(min_height=416, min_width=416, border_mode=0, value=(0.5,0.5,0.5), p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0, 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0, 
            min_visibility=0,
            label_fields=['labels']
        )
    )
