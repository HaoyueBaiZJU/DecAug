import os
from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data as data
from PIL import Image
import numpy as np



def get_train_loaders(txt_pth_list, args):
    if args.nico_cls == "animal":
        _mean = np.array([x / 255.0 for x in [104.079, 107.423, 104.984]])
        _std =  np.array([x / 255.0 for x in [47.554, 48.668, 53.170]])
    elif args.nico_cls == "vehicle":
        _mean = np.array([x / 255.0 for x in [159.063, 155.365, 154.892]])
        _std =  np.array([x / 255.0 for x in [56.217, 55.824, 53.863]])
    transf = [
        transforms.RandomResizedCrop(84),
        #transforms.Resize((84, 84)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize(_mean,_std )
    ]
        # concat_dataset = ConcatDataset(dataset_list)
    
    dataloader_list = []
    for txt_pth in txt_pth_list:
        dataset = NicoDataset(txt_pth,args,  transforms.Compose(transf))
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                shuffle=True, num_workers=4, pin_memory=True)
        dataloader_list.append(dataloader)
    return dataloader_list


def get_val_loader(txt_pth, args):
    
    if args.nico_cls == "animal":
        _mean = np.array([x / 255.0 for x in [104.079, 107.423, 104.984]])
        _std =  np.array([x / 255.0 for x in [47.554, 48.668, 53.170]])
    elif args.nico_cls == "vehicle":
        _mean = np.array([x / 255.0 for x in [159.063, 155.365, 154.892]])
        _std =  np.array([x / 255.0 for x in [56.217, 55.824, 53.863]])
    transf = [
        transforms.Resize(84+8),
        transforms.CenterCrop(84),
        transforms.ToTensor(),
        transforms.Normalize(_mean, _std)
    ]
        # concat_dataset = ConcatDataset(dataset_list)
    
    dataset = NicoDataset(txt_pth,args, transforms.Compose(transf))
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)
    return dataloader



class NicoDataset(data.Dataset):
    def __init__(self, csv_path, args, img_transform = None):
        self.data_root = "/home/ma-user/work/OOD/data/nico/{}/images/".format(args.nico_cls)
        self.csv_path = csv_path
        self.img_transform = img_transform
        self.data, self.label, self.concept = self.parse_csv()

    def __getitem__(self, i):
        data, label, concept = self.data[i], self.label[i], self.concept[i]
        with open(data, 'rb') as f:
            image = Image.open(f).convert('RGB')
            image = self.img_transform(image)
        return image, label


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
        concept = [concept_to_idx[x] for x in concept_name]
        print("class_to_idx:",class_to_idx)
        return data, label, concept
    
    
    def __len__(self):
        return len(self.data)

