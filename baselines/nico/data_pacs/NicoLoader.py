import os
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


class NicoDataset(data.Dataset):
    def __init__(self, dataroot,csv_path, jig_classes=23,img_transformer=None,tile_transformer=None, patches=True, bias_whole_image=None):
        self.data_root = dataroot
        self.csv_path = csv_path
        self.bias_whole_image = bias_whole_image
        self.permutations = self.__retrieve_permutations(jig_classes)
        self._augment_tile = tile_transformer
        self.img_pth, self.labels = self.parse_csv()
        self.grid_size = 3
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                return torchvision.utils.make_grid(x, self.grid_size, padding=0)
            self.returnFunc = make_grid
        self._transform = img_transformer
    
    def parse_csv(self):
        # pdb.set_trace()
        with open(self.csv_path, 'r') as csvfile:
            # with mox.file.File(csv_path, 'r') as csvfile:
            lines = [x.strip() for x in csvfile.readlines()]

        data = []
        label_name = []
        concept_name = []

        for l in lines:
            name, wnid, conid = l.split(',')
            path = os.path.join(self.data_root, name)
            path = path.replace('\\', '/')
            data.append(path)
            label_name.append(wnid)
            concept_name.append(conid)
        classes = list(set(label_name))
        classes.sort()
        concepts = list(set(concept_name))
        concepts.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        concept_to_idx = {concepts[j]: j for j in range(len(concepts))}
        label = [class_to_idx[x] for x in label_name]
        print("class_to_idx:",class_to_idx)
        return data, label
    
    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self._augment_tile(tile)
        return tile
    
    def get_image(self, index):
        framename = self.img_pth[index]
        img = Image.open(framename).convert('RGB')
        return self._transform(img)
        
    def __getitem__(self, index):
        img = self.get_image(index)
        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(img, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if self.bias_whole_image:
            if self.bias_whole_image > random():
                order = 0
        if order == 0:    
            data = tiles
        else:
            #print(len(self.permutations))
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
        data = torch.stack(data, 0)
        #print("data shape:",data.shape)
        return self.returnFunc(data), int(order), int(self.labels[index])

    def __len__(self):
        return len(self.img_pth)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

class NicoTestDataset(NicoDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.img_pth[index]
        img = Image.open(framename).convert('RGB')
        return self._transform(img),0, int(self.labels[index])
