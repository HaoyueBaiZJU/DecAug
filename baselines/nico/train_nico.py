import time
import os
import numpy as np
import torch
import torch.utils.data as data
import datasets
import hparams_registry
import torchvision
import torchvision.transforms as transforms
from dataloader import get_train_loaders, get_val_loader
from torch.utils.data import DataLoader,ConcatDataset
from PIL import Image
from torch import nn, optim, autograd
import torchvision.models as models
import argparse
import logging
import warnings
import algorithms_img as algorithms
from algorithms_img import *
import misc
warnings.filterwarnings('ignore')
import numpy as np

parser = argparse.ArgumentParser(description='nico')

parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--dataset', type=str, default='NICO')
parser.add_argument('--algorithm', type=str, default='ERM')
parser.add_argument('--nico_cls', type=str, default='animal')
parser.add_argument('--data_root', type=str, default='/home/ma-user/work/OOD/data/nico/')

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

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class((3,84,84),args.num_classes,3, hparams)

    print("network:",algorithm)
    algorithm.cuda()
    source_domain = [args.nico_cls + "_domain_{}.csv".format(i) for i in range(1,4) ]
    val_domain = args.nico_cls + "_domain_val.csv"
    target_domain = args.nico_cls + "_domain_4.csv"
    print("source_domain {},val_domain {},target_domain {}".format(source_domain,val_domain,target_domain))
    train_loaders = get_train_loaders([os.path.join(args.data_root, s) for s in source_domain], args)
    val_loader = get_val_loader(os.path.join(args.data_root,val_domain),args)
    test_loader = get_val_loader(os.path.join(args.data_root, target_domain), args)
    
    train_steps = max([len(d) for d in train_loaders])
    print("train steps:",train_steps) 
    val_accs = []
    test_accs = []
    best_val_acc = 0.0
    best_epoch = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        train(train_loaders, algorithm, device,train_steps)
        val_acc = infer(val_loader, algorithm, epoch, "val")        
        test_acc = infer(test_loader, algorithm, epoch, "test") 
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
    print('Best val_acc: %f'% best_val_acc + " best epoch: {}".format(best_epoch))
    print("the chosen test acc: ",test_accs[best_epoch])


def train(train_loaders, algorithm, device,train_steps): 
    algorithm.train()
    loaders = train_loaders
    train_minibatches_iterator = zip(*loaders)
    for step in range(train_steps):
        for i in range(3):
            if step >0 and step % len(train_loaders[i]) == 0:
                #print("change _ th loader:",i)
                loaders[i] = train_loaders[i]
                train_minibatches_iterator = zip(*loaders)
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
    print( 'epoch:[{}]'.format(epoch),val_test + ' Accuracy of the network on the images: %d %%' % (
            100 * test_acc))
    return test_acc

if __name__ == '__main__':
    main()

