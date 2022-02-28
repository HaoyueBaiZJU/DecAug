from torchvision import datasets
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data as data
from PIL import Image


def get_train_loader(txt_pth_list, args):
    transf = [
        transforms.RandomResizedCrop((222, 222), (0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    # concat_dataset = ConcatDataset(dataset_list)
    dataset_list = []
    for txt_pth in txt_pth_list:
        dataset = NicoDataset(txt_pth, transforms.Compose(transf))
        dataset_list.append(dataset)
    dataset_all = ConcatDataset(dataset_list)
    dataloader = DataLoader(dataset_all, batch_size=args.batch_size,
                            shuffle=True, num_workers=4, pin_memory=True)
    return dataloader


def get_val_loader(txt_pth, args):
    transf = [
        transforms.Resize((222, 222)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    # concat_dataset = ConcatDataset(dataset_list)

    dataset = NicoDataset(txt_pth, transforms.Compose(transf))
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    return dataloader


class NicoDataset(data.Dataset):
    def __init__(self, txt_pth, img_transform=None):
        self.txt_path = txt_pth
        self.img_transform = img_transform
        self.read_txt()

    def read_txt(self):
        self.img_pth, self.context, self.labels = [], [], []
        with open(self.txt_path) as f:
            for line in f.readlines():
                self.img_pth.append(line.split(",")[0])
                self.context.append(line.split(",")[1])
                self.labels.append(line.split(",")[2].split("\n")[0])

    def __getitem__(self, index):
        framename = self.img_pth[index]
        img = Image.open(framename).convert('RGB')
        return self.img_transform(img), int(self.labels[index])

    def __len__(self):
        return len(self.img_pth)

