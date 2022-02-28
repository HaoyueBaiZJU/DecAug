import time
import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim, autograd
from random import sample, random
import torchvision.models as models
from dataloader import get_train_loader,get_val_loader
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='NICO')

parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--data_root', type=str, default='./data')
parser.add_argument('--nico_cls', type=str, default='animal')
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--optimizer', type=str, default='sgd')
args, unparsed = parser.parse_known_args()

def main():
    source_domain = [args.nico_cls + "_domain_" + str(i) + ".txt" for i in range(1, 4)]
    target_domain = args.nico_cls + "_domain_4.txt"
    train_loader = get_train_loader([os.path.join(args.data_root, s) for s in source_domain], args)
    test_loader = get_val_loader(os.path.join(args.data_root, target_domain), args)
    resnet18 = models.resnet18(pretrained=False)
    model = resnet18
    model.load_state_dict(torch.load('./resnet18.pth'), strict=False)
    model.fc = nn.Linear(512, args.num_classes)
    model = model.cuda()
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss().cuda()
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.6 * args.epochs)], 0.1)
    # scheduler = utils.WarmupCosine(optimizer, args.epochs * len(train_loader),
    #                                2 * len(train_loader))
    best_acc = 0.0
    train_accs = []
    test_accs = []
    best_accs = []
    for epoch in range(args.epochs):
        start_time = time.time()
        train_acc = train(train_loader, model, criterion, optimizer, epoch)
        valid_acc = infer(test_loader, model, criterion, epoch)
        if valid_acc > best_acc:
            best_acc = valid_acc
        logging.info('Valid_acc: %f', valid_acc)
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration)
        train_accs.append(train_acc)
        test_accs.append(valid_acc)
        best_accs.append(best_acc)
    print('Best Valid_acc: %f'% best_acc)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    correct = 0
    total = 0
    for step,(input, target) in enumerate(train_loader):
        if step % 10 == 0:
            print("epoch:{} step: {}/{}".format(epoch,step,len(train_loader)))
        optimizer.zero_grad()
        input = input.cuda()
        target = target.cuda()
        logits = model(input)
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        _, predicted = torch.max(logits, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    train_acc = correct / total
    print('Accuracy of the network on the training images: %d %%' % (
            100 * train_acc))
    return train_acc

def infer(valid_queue, model, criterion, epoch):
    model.eval()
    correct = 0
    total = 0
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            logits = model(input)

        _, predicted = torch.max(logits, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    test_acc = correct / total
    print('Accuracy of the network on the test images: %d %%' % (
            100 * test_acc))
    return test_acc

if __name__ == '__main__':
    main()
