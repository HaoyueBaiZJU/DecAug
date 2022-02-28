from os.path import join, dirname
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.NicoLoader import NicoDataset, NicoTestDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset

available_datasets = ['animal_domain_1','animal_domain_2','animal_domain_3','animal_domain_4',\
                        'animal_domain_val','vehicle_domain_val',\
                       'vehicle_domain_1','vehicle_domain_2','vehicle_domain_3','vehicle_domain_4']

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args,patches):
    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    dataroot = "/home/ma-user/work/OOD/data/nico/{}/images/".format(args.nico_cls)
    csv_path = "/home/ma-user/work/OOD/data/nico/"
    img_transformer, tile_transformer = get_train_transformers(args)
    for dname in dataset_list:
        #name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'color-mnist', '%s.txt' % dname), args.val_size)
        train_dataset = NicoDataset(dataroot,csv_path+dname+'.csv', jig_classes=args.jigsaw_n_classes,img_transformer=img_transformer,tile_transformer=tile_transformer,patches=patches, bias_whole_image=args.bias_whole_image)
        datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    return loader


def get_val_dataloader(args, dname, patches=False):
    dataroot = "/home/ma-user/work/OOD/data/nico/{}/images/".format(args.nico_cls)
    csv_path = "/home/ma-user/work/OOD/data/nico/"
    img_transformer = get_val_transformer(args)
    val_dataset = NicoTestDataset(dataroot, csv_path+dname+'.csv',jig_classes=args.jigsaw_n_classes,img_transformer=img_transformer, patches=patches)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    return loader


def get_jigsaw_val_dataloader(args, patches=False):
    names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    img_tr = [transforms.Resize((args.image_size, args.image_size))]
    tile_tr = [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    img_transformer = transforms.Compose(img_tr)
    tile_transformer = transforms.Compose(tile_tr)
    val_dataset = JigsawDataset(names, labels, patches=patches, img_transformer=img_transformer,
                                      tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    if args.limit_target and len(val_dataset) > args.limit_target:
        val_dataset = Subset(val_dataset, args.limit_target)
        print("Using %d subset of val dataset" % args.limit_target)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
    if args.nico_cls == "animal":
        _mean = np.array([x / 255.0 for x in [104.079, 107.423, 104.984]])
        _std =  np.array([x / 255.0 for x in [47.554, 48.668, 53.170]])
    elif args.nico_cls == "vehicle":
        _mean = np.array([x / 255.0 for x in [159.063, 155.365, 154.892]])
        _std =  np.array([x / 255.0 for x in [56.217, 55.824, 53.863]])
    img_tr = [transforms.RandomResizedCrop(args.image_size),transforms.RandomHorizontalFlip()]
    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize(_mean,_std)]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    if args.nico_cls == "animal":
        _mean = np.array([x / 255.0 for x in [104.079, 107.423, 104.984]])
        _std =  np.array([x / 255.0 for x in [47.554, 48.668, 53.170]])
    elif args.nico_cls == "vehicle":
        _mean = np.array([x / 255.0 for x in [159.063, 155.365, 154.892]])
        _std =  np.array([x / 255.0 for x in [56.217, 55.824, 53.863]])    
    img_tr = [transforms.Resize(args.image_size+8),transforms.CenterCrop(args.image_size), transforms.ToTensor(),
              transforms.Normalize(_mean,_std)]
    return transforms.Compose(img_tr)


def get_target_jigsaw_loader(args):
    img_transformer, tile_transformer = get_train_transformers(args)
    name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
    dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
                            tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader
