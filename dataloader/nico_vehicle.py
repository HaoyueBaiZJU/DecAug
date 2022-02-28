import torch
import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

import os
import moxing as mox


ROOT_PATH = "/home/ma-user/work/dataset"
IMAGE_PATH1 = osp.join(ROOT_PATH, 'NicoNew/vehicle/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'NicoNew/vehicle/split_v2_ood')


class NicoVehicle(Dataset):
    """ Usage:
    """
    def __init__(self, setname, args, augment=False):
        csv_path = osp.join(SPLIT_PATH, setname + '.csv')

        self.data, self.label, self.concept = self.parse_csv(csv_path, setname)
        self.num_class = len(set(self.label))
        self.num_concept = len(set(self.concept))
        image_size = args.image_size
        

        if augment:
            transforms_list = [
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    ]
        else:
            transforms_list = [
                    transforms.Resize(image_size + 8),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    ]
        
        self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([x / 255.0 for x in [159.063, 155.365, 154.892]]),
                                         np.array([x / 255.0 for x in [56.217, 55.824, 53.863]]))

                    ]
                )
            
    def parse_csv(self, csv_path, setname):
        with open(csv_path, 'r') as csvfile:
            lines = [x.strip() for x in csvfile.readlines()]
        

        data = []
        label_name = []
        concept_name = []

        for l in tqdm(lines, ncols=64):
            name, wnid, conid = l.split(',')
            path = osp.join(IMAGE_PATH1, name)
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
        concept = [concept_to_idx[x] for x in concept_name]
            
        return data, label, concept



    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label, concept = self.data[i], self.label[i], self.concept[i]
        with open(data, 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = self.transform(image)

        return image, label, concept


