import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import matplotlib.pyplot as plt

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


class CmnistDataset(data.Dataset):
    def __init__(self, txt_path,dataroot, jig_classes=23, patches=True,bias_whole_image=None):
        self.txt_path = txt_path
        self.dataroot = dataroot
        self.bias_whole_image = bias_whole_image
        self.permutations = self.__retrieve_permutations(jig_classes)
        self.read_txt()
        self.grid_size = 2
        if patches:
            self.returnFunc = lambda x: x
        else:
            def make_grid(x):
                #add a new dim
                z = torch.ones(4,3,7,7)*1.5
                z[:,0,:,:] = x[:,0,:,:]
                z[:,1,:,:] = x[:,1,:,:]
                y = torchvision.utils.make_grid(z, self.grid_size, padding=0)
                res = y[0:2,:,:]
                return res
            self.returnFunc = make_grid
    
    def read_txt(self):
        with open(self.txt_path) as f:
            file_path = f.readlines()[0]
        file_path = os.path.join(self.dataroot, file_path)
        dataset = np.load(file_path, allow_pickle=True)
        self.images = torch.from_numpy(dataset.item()['images'])
        self.labels = torch.from_numpy(np.uint8(dataset.item()['labels'].reshape(-1,)))

    def get_tile(self, img, n):
        w = int(float(img.shape[1]) / self.grid_size)
        y = int(n / self.grid_size)
        x = int(n % self.grid_size)
        #print("w,x,y,n:",w,x,y,n)
        tile = img[:,x * w : (x + 1) * w, y * w: (y + 1) * w]
        return tile
    
    def get_image(self, index):
        img = self.images[index]
        return img
        
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
        #print(img.shape)
        #img = img[1,:,:]
        #plt.imsave("./digit.jpg",img)
        res = self.returnFunc(data)
        #plt.imsave("./jig.jpg",res[1,:,:])
        #print("res:",res)
        return res, int(order), int(self.labels[index])

    def __len__(self):
        return len(self.images)

    def __retrieve_permutations(self, classes):
        all_perm = np.load('permutations_%d.npy' % (classes))
        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm


class CmnistTestDataset(CmnistDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        return self.images[index], 0, int(self.labels[index])


