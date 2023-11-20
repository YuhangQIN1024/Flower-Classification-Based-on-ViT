from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform
        delete_img = []
        for index, img_path in tqdm(enumerate(images_path)):
            img = Image.open(img_path)
            w, h = img.size
            ratio = w / h
            if ratio > 10 or ratio < 0.1:
                delete_img.append(index)
        for index in delete_img[::-1]:
            self.images_path.pop(index)
            self.images_class.pop(index)

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels