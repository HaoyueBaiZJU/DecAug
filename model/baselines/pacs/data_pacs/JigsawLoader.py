import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class JigsawDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None):
        self.data_path = ""
        self.names = names
        self.labels = labels
        self.N = len(self.names)       
        self._image_transformer = img_transformer
        
    def get_image(self, index):
        framename = self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        return img, int(self.labels[index])

    def __len__(self):
        return len(self.names)

class JigsawTestDataset(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        # print("self.data_path:{},framename:{}".format(self.data_path,framename))
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index])


class JigsawTestDatasetMultiple(JigsawDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        _img = Image.open(framename).convert('RGB')
        img = self._image_transformer(_img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        images = []
        jig_labels = []
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile
        for order in range(0, len(self.permutations)+1, 3):
            if order==0:
                data = tiles
            else:
                data = [tiles[self.permutations[order-1][t]] for t in range(n_grids)]
            data = self.returnFunc(torch.stack(data, 0))
            images.append(data)
            jig_labels.append(order)
        images = torch.stack(images, 0)
        jig_labels = torch.LongTensor(jig_labels)
        return images, jig_labels, int(self.labels[index])
