from os.path import join, dirname

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data import StandardDataset
from data.CmnistLoader import CmnistDataset, CmnistTestDataset, get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset

available_datasets = ['0.1','0.2','0.3','0.9']

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
    dataroot = "/home/ma-user/work/OOD/data/color-mnist/"
    img_transformer, tile_transformer = get_train_transformers(args)
    for dname in dataset_list:
        #name_train, name_val, labels_train, labels_val = get_split_dataset_info(join(dirname(__file__), 'color-mnist', '%s.txt' % dname), args.val_size)
        train_dataset = CmnistDataset(dataroot+dname+'.txt',dataroot, jig_classes=args.jigsaw_n_classes,patches = patches, bias_whole_image  =args.bias_whole_image)
        datasets.append(train_dataset)
    dataset = ConcatDataset(datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    return loader


def get_val_dataloader(args, patches=False):
    dataroot = "/home/ma-user/work/OOD/data/color-mnist/"
    val_dataset = CmnistTestDataset(dataroot+args.target+'.txt',dataroot, jig_classes=args.jigsaw_n_classes,patches=patches)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
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
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)


def get_target_jigsaw_loader(args):
    img_transformer, tile_transformer = get_train_transformers(args)
    name_train, _, labels_train, _ = get_split_dataset_info(join(dirname(__file__), 'txt_lists', '%s_train.txt' % args.target), 0)
    dataset = JigsawDataset(name_train, labels_train, patches=False, img_transformer=img_transformer,
                            tile_transformer=tile_transformer, jig_classes=args.jigsaw_n_classes, bias_whole_image=args.bias_whole_image)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    return loader
