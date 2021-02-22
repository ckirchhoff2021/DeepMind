from PIL import Image
from torchvision import datasets


class CIFAR10Instance(datasets.CIFAR10):
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        img1 = img
        img2 = img
        
        if self.transform is not None:
            img1 = self.transform(img)
            img2 = self.transform(img)
          
        return img1, img2, target