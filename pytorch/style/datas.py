import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


content_folder = 'res/content_resized'
style_folder = 'res/style_resized'

image_transform = transforms.Compose([
    transforms.RandomCrop(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class StyleDataset(Dataset):
    def __init__(self):
        super(StyleDataset, self).__init__()
        self.contents = []
        self.styles = []
        self.initialize()

    def initialize(self):
        content_images = os.listdir(content_folder)
        style_images = os.listdir(style_folder)
        for content in content_images:
            if not content.endswith('.jpg'):
                continue
            content_data = Image.open(os.path.join(content_folder, content))
            content_tensor = image_transform(content_data)
            for style in style_images:
                if not style.endswith('.jpg'):
                    continue
                style_data = Image.open(os.path.join(style_folder, style))
                style_tensor = image_transform(style_data)
                self.contents.append(content_tensor)
                self.styles.append(style_tensor)

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self.contents[index], self.styles[index]


if __name__ == '__main__':
    style_datas = StyleDataset()
    print(len(style_datas))







