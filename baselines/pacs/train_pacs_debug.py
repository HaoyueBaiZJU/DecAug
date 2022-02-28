import time
import os
import numpy as np
import torch
from data_pacs import data_helper
from data_pacs.data_helper import available_datasets
import torch.utils.data as data
from torch.utils.data import TensorDataset
import datasets
import hparams_registry
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,ConcatDataset
from PIL import Image
from torch import nn, optim, autograd
from random import sample, random
import torchvision.models as models
import argparse
import logging
import warnings
import algorithms_img as algorithms
from algorithms_img import *
import misc
warnings.filterwarnings('ignore')
import numpy as np
from fast_data_loader import InfiniteDataLoader, FastDataLoader

parser = argparse.ArgumentParser(description='cmnist')

parser.add_argument('--num_classes', type=int, default=1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument("--image_size", type=int, default=224, help="Image size")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
parser.add_argument("--target", choices=available_datasets, help="Target")
parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
parser.add_argument("--jitter", default=0.4, type=float, help="Color jitter amount")
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--dataset', type=str, default='PACS')
parser.add_argument('--algorithm', type=str, default='ERM')
parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
parser.add_argument('--data_dir', type=str, default='/home/ma-user/work/OOD/data/')
args, unparsed = parser.parse_known_args()


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('using gpu:', os.environ['CUDA_VISIBLE_DEVICES'])
    print('running algorithm:',args.algorithm)
    hparams = hparams_registry.default_hparams(args.algorithm,args.dataset)
    print(hparams)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            [0], hparams)
    else:
        raise NotImplementedError
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - 1, hparams)
        
    print("network:",algorithm)
    algorithm.cuda()
  
    train_datasets, val_datasets = data_helper.get_train_dataset(args)         
    target_dataset = data_helper.get_val_dataset(args)  
    val_dataset = ConcatDataset(val_datasets)
    train_dataset = ConcatDataset(train_datasets)
    train_steps = int(len(train_dataset)/args.batch_size )
    print("train steps:",train_steps)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    test_loader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    import pdb
    #pdb.set_trace()
    train_accs = []
    val_accs = []
    test_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        train(train_loader, algorithm, device,train_steps)
        val_acc = infer(val_loader, algorithm, epoch, "val")        
        test_acc = infer(test_loader, algorithm, epoch, "test") 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        end_time = time.time()
        duration = end_time - start_time
        if (epoch) %100 == 0:
            print('Epoch time: %ds.' % duration)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    print('Best val_acc: %f'% best_val_acc + " best epoch: {}".format(best_epoch))
    print("the chosen test acc: ",test_accs[best_epoch])

def train(train_loader, algorithm, device,train_steps): 
    algorithm.train()
    loader = train_loader
    train_minibatches_iterator = zip(*[loader])
    for step in range(train_steps):
        minibatches_device = [(x.to(device), y.to(device)) for x,y in next(train_minibatches_iterator)]
        loss = algorithm.update(minibatches_device)

def infer(valid_queue, algorithm, epoch, val_test):
    algorithm.eval()
    correct = 0
    total = 0
    for input, target in valid_queue:
        input = input.cuda()
        target = target.cuda()
        logits = algorithm.predict(input)
        _, predicted = torch.max(logits, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    test_acc = correct / total
    print( 'epoch:[{}]'.format(epoch),val_test + ' Accuracy of the network on the images: %.2f %%' % (
            100 * test_acc))
    return test_acc

if __name__ == '__main__':
    main()

