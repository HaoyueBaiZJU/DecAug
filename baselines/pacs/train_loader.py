import time
import os
import numpy as np
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset
import hparams_registry
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
from torch import nn, optim, autograd
from random import sample, random
import torchvision.models as models
from torchvision import datasets
import argparse
import logging
import warnings
import algorithms
from algorithms import *
import torchattacks
#atk = torchattacks.PGD(model, eps = 4/255, alpha = 2/255, steps=4)
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser(description='cmnist')

parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=20000)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--algorithm', type=str, default='ERM')
parser.add_argument('--dataset', type=str, default='ColoredMNIST')
args, unparsed = parser.parse_known_args()


def create_cmnist():
    mnist = datasets.MNIST('/home/ma-user/work/OOD/data/', train=True, download=False)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments
    def make_environment(images, labels, e, label_noise):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        #images = images.reshape((-1, 28, 28))
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(label_noise, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        x = (images.float() / 255.).cuda()
        import pdb
        #pdb.set_trace()
        #y = labels[:, None].cuda()
        y = labels.cuda()
        y = y.to(device=y.get_device(), dtype=torch.int64)
        return (x, y)

    envs = [make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1, 0.25),
    make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.2, 0.25),
    make_environment(mnist_val[0], mnist_val[1], 0.9, 0.25)]
    return envs

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    print('using gpu:', os.environ['CUDA_VISIBLE_DEVICES'])
    print('running algorithm:',args.algorithm)
    hparams = hparams_registry.default_hparams(args.algorithm,args.dataset)
    hparams['batch_size'] = 128
    print(hparams)
    final_test_accs = []
    for restart in range(3):
        algorithm_class = algorithms.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class((2,14,14), args.num_classes,2, hparams)
        algorithm.cuda()
        print("network:",algorithm)
        envs = create_cmnist()
        train_envs = envs[0:2]
        test_env = envs[2]
        train_loaders = [DataLoader(TensorDataset(env[0],env[1]),\
        batch_size = hparams['batch_size'],shuffle=True) for env in train_envs]
        test_loader = DataLoader(TensorDataset(test_env[0],test_env[1]),\
        batch_size = hparams['batch_size'],shuffle=False)
        train_accs = []
        test_accs = []
        for epoch in range(args.epochs):
            start_time = time.time()
            #train_acc = train(train_envs, algorithm, epoch)
            #test_acc = infer(test_env, algorithm, epoch, "test")
            train_acc = train_with_loader(train_loaders, algorithm, epoch)
            test_acc = infer(test_env, algorithm, epoch, "test")
            end_time = time.time()
            duration = end_time - start_time
            if (epoch+1) %100 == 0:
                print('Epoch time: %ds.' % duration)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        print("the last test acc:{} ".format(test_accs[-1]))
        #final_test_accs.append(test_accs[-1].detach().cpu().numpy())
        final_test_accs.append(test_accs[-1])
    print("mean/std of test acc: {}/{}".format(np.mean(final_test_accs), np.std(final_test_accs)))

def train(train_envs, algorithm, epoch):
    import pdb
    #minibatches_device = [(x, y) for x,y in zip(train_envs)]
    loss,train_acc = algorithm.update(train_envs)
    if (epoch+1) %100 == 0:
        print('epoch:[{}]'.format(epoch), 'Accuracy of the network on the training images: %d %%' % (
            100 * train_acc))
    return train_acc

def train_with_loader(train_loaders, algorithm, epoch):
    train_minibatches_iterator = zip(*train_loaders)
    steps = min([len(loader) for loader in train_loaders])
    for _ in range(steps):
        minibatches_device = [(x, y) for x,y in next(train_minibatches_iterator)]
        loss,train_acc = algorithm.update(minibatches_device)
    return train_acc

def infer(test_env, algorithm, epoch, val_test):
    import pdb
    #pdb.set_trace()
    logits = algorithm.predict(test_env[0])
    test_acc = algorithms.mean_accuracy(logits,test_env[1])  
    if (epoch+1) %1 == 0:
        print( 'epoch:[{}]'.format(epoch),val_test + ' Accuracy of the network on the images: %d %%' % (
            100 * test_acc))
    return test_acc

if __name__ == '__main__':
    main()

