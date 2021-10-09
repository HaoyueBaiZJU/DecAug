import numpy as np
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from random import sample, random


from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset


import os.path as osp
from torch.utils.data import Dataset
from tqdm import tqdm

import os

from random import sample, random



#ROOT_PATH= "./OOD/data"
ROOT_PATH= "/home/ma-user/work/OOD/data"



IMAGE_PATH1 = os.path.join(ROOT_PATH, 'pacs/kfold')
SPLIT_PATH = os.path.join(ROOT_PATH, 'pacs/kfold')


class pacsDataset(data.Dataset):
    def __init__(self, setname, args):

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


        if setname != 'test':
            import pdb
            domain = ['cartoon', 'art_painting', 'photo', 'sketch']
            domain.remove(args.targetdomain)

            fulldata = []
            label_name = []
            fullconcept = []
            i = 0
            for domain_name in domain:
                txt_path = os.path.join(SPLIT_PATH, domain_name + '.txt')

                images, labels = self._dataset_info(txt_path)
                concept = [i] * len(labels)
                fulldata.extend(images)
                label_name.extend(labels)
                fullconcept.extend(concept)
                i += 1

            classes = list(set(label_name))
            classes.sort()

            class_to_idx = {classes[i]: i for i in range(len(classes))}

            fulllabel = [class_to_idx[x] for x in label_name]

            name_train, name_val, labels_train, labels_val, concepts_train, concepts_val = self.get_random_subset(fulldata, fulllabel, fullconcept, 0.1)
            if setname == "train":
                self.data = name_train
                self.label = labels_train
                self.concept = concepts_train
            else:
                self.data = name_val
                self.label = labels_val
                self.concept = concepts_val


        else:
            domain = args.targetdomain
            txt_path = os.path.join(SPLIT_PATH, domain + '.txt')
            self.data, self.label = self._dataset_info(txt_path)
            self.concept = [0] * len(self.data)

        self.num_class = np.max(self.label) + 1
        self.num_concept = np.max(self.concept) + 1


    def _dataset_info(self, txt_labels):
        with open(txt_labels, 'r') as f:
            images_list = f.readlines()

        file_names = []
        labels = []
        for row in images_list:
            row = row.split(' ')
            path = os.path.join(IMAGE_PATH1, row[0])
            path = path.replace('\\', '/')

            file_names.append(path)
            labels.append(int(row[1]))

        return file_names, labels

    def get_random_subset(self, names, labels, concepts, percent):
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
        concepts_val = [concepts[k] for k in random_index]
        concepts_train = [v for k, v in enumerate(concepts) if k not in random_index]
        return name_train, name_val, labels_train, labels_val, concepts_train, concepts_val


    def __getitem__(self, index):
        data, label, concept = self.data[index], self.label[index], self.concept[index]

        _img = Image.open(data).convert('RGB')
        img = self._image_transformer_full(_img)

        return img, label, concept

    def __len__(self):
        return len(self.data)

