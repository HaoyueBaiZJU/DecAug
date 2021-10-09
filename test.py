import argparse
import os.path as osp
import numpy as np
import torch
from torch import nn, optim, autograd

import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.utils import pprint, set_gpu, ensure_path, AverageMeter, Timer, accuracy, one_hot
from tensorboardX import SummaryWriter
import time
import datetime

from keras.utils import to_categorical

import random

from torch.autograd import Variable

import os
import logging
import sys

'''Train Benchmark'''    

def get_args():
    parser = argparse.ArgumentParser()
    # Basic Parameters
    parser.add_argument('--dataset', type=str, default='pacs', 
                       choices=['pacs'])
    parser.add_argument('--backbone_class', type=str, default='resnet18', choices=['resnet18'])   

    # Optimization Parameters
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.05)  
    parser.add_argument('--init_weights', type=str, default=None)    
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--image_size', type=int, default=225)
    parser.add_argument('--prefetch', type=int, default=16)

    # Model Parameters
    parser.add_argument('--model_type', type=str, 
                       choices=['DecAug'], default='DecAug')
    parser.add_argument('--balance1', type=float, default=0.01)  # the balance parameters for category    
    parser.add_argument('--balance2', type=float, default=0.01)  # the balance parameters for context    
    parser.add_argument('--balanceorth', type=float, default=0.01) # the balance parameters for orth
    parser.add_argument('--perturbation', type=float, default=1)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--targetdomain', type=str, default='photo', choices=['cartoon', 'art_painting', 'photo', 'sketch'])
    parser.add_argument('--pretrain', type=bool, default=True)

    # Other Parameters
    parser.add_argument('--gpu', default='0')

    args, unknown_args = parser.parse_known_args()    

    set_gpu(args.gpu)
    
    args.concept_path = './saves'
    args.init_path = os.path.join('./saves/initialization', 'resnet18.pth')

    args.save_path = os.path.join('./exp_log', args.targetdomain)

    return args

def get_model(args, s1_data):
    if args.model_type == 'DecAug':
        from model.models.DecAug import bgor2
        model = bgor2(args)
    else:
        raise ValueError('No Such Model')
        
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    return model

def get_optimizer(args, model):
    parameters = model.named_parameters()
    top_list, bottom_list = [], []
    for k, v in parameters:
        if 'encoder' in k:
            bottom_list.append(v)
        else:
            top_list.append(v)
            
    if args.warmup > 0:
        optimizer_warmup = torch.optim.SGD(top_list, 
                                           lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)   
        lr_scheduler_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer_warmup, lambda epoch: epoch * (1/args.warmup))               
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)   
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)  
        return optimizer, lr_scheduler, optimizer_warmup, lr_scheduler_warmup
    else:              
        optimizer = torch.optim.SGD([{'params': top_list, 'lr': args.lr * 10},
                                     {'params':bottom_list}],
                                    lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)   
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=args.gamma)  
        return optimizer, lr_scheduler, None, None

def get_loader(args):
    
    if args.dataset == 'pacs':
        from model.dataloader.pacsLoader import pacsDataset as Dataset
        trainset = Dataset('train', args)
        valset = Dataset('val', args)
        testset = Dataset('test', args)

    else:
        raise ValueError('Non-supported Dataset.')

    args.num_class = trainset.num_class
    args.num_concept = trainset.num_concept
    
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.prefetch, pin_memory=True)
    val_loader = DataLoader(dataset=valset, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
    test_loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False, num_workers=args.prefetch, pin_memory=True)
  
    return train_loader, test_loader, test_loader

    

def test_model(args, model, test_loader, s1_data):
    model.load_state_dict(torch.load(osp.join(args.save_path, 'max_acc.pth'))['params'])

    t1 = AverageMeter()
    ta = AverageMeter()
    tv = AverageMeter()
    tc = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            if torch.cuda.is_available():
                data, gt_label, gt_concept = [_.cuda() for _ in batch]
            else:
                data, gt_label, gt_concept = batch[0], batch[1], batch[2]
   
            if args.model_type == 'DecAug':

                logits, logits_category, logits_concept, feature, feature_category, feature_concept = model(data)
                loss1 = F.cross_entropy(logits_category, gt_label)
                loss2 = F.cross_entropy(logits_concept, gt_concept)

                parm = {}
                for name, parameters in model.named_parameters():
                    parm[name] = parameters
                # concept branch
                w_branch = parm['concept_branch.weight']
                w_tensor = parm['fcc0.weight']
                b_tensor = parm['fcc0.bias']
                # classification branch
                w_branch_l = parm['category_branch.weight']
                w_tensor_l = parm['fc0.weight']
                b_tensor_l = parm['fc0.bias']

                w_out = parm['classification.weight']
                b_out = parm['classification.bias']

                w = torch.matmul(w_tensor, w_branch)
                grad = -1 * w[gt_concept] + torch.matmul(logits_concept.detach(), w)
                grad_norm = grad / (grad.norm(2, dim=1, keepdim=True) + args.epsilon)

                w_l = torch.matmul(w_tensor_l, w_branch_l)
                grad_l = -1 * w_l[gt_label] + torch.matmul(logits_category.detach(), w_l)
                grad_norm_l = grad_l / (grad_l.norm(2, dim=1, keepdim=True) + args.epsilon)
                b, L = grad_norm_l.shape

                grad_norm = grad_norm.reshape(b, 1, L)
                grad_norm_l = grad_norm_l.reshape(b, L, 1)
                loss_orth = ((torch.bmm(grad_norm, grad_norm_l).cuda()) ** 2).sum()

                grad_aug = -1 * w_tensor[gt_concept] + torch.matmul(logits_concept.detach(), w_tensor)
                FGSM_attack = args.perturbation * (grad_aug.detach() / (grad_aug.detach().norm(2, dim=1, keepdim=True) +  args.epsilon))
                ratio = random.random()

                feature_aug = ratio * FGSM_attack
                embs = torch.cat((feature_category, feature_concept), 1)
                output = torch.matmul(embs, w_out.transpose(0, 1)) + b_out
                logits_class = output

                loss_class = F.cross_entropy(logits_class, gt_label)
                loss = loss_class + args.balance1 * loss1 + args.balance2 * loss2 + args.balanceorth * loss_orth

            else:
                raise ValueError('')

            acc = accuracy(logits_class.data, gt_label.data, topk=(1,))[0]
            acc_concept = accuracy(logits_concept.data, gt_concept.data, topk=(1,))[0]
            acc_category = accuracy(logits_category.data, gt_label.data, topk=(1,))[0]
      
            t1.update(loss.item(), data.size(0))
            ta.update(acc.item(), data.size(0))
            tv.update(acc_concept.item(), data.size(0))
            tc.update(acc_category.item(), data.size(0))          
    
    t1 = t1.avg
    ta = ta.avg
    tv = tv.avg
    tc = tc.avg

    logging.info('Test acc={:.4f}, acc_concept={:.4f}, acc_category={:.4f}'.format(ta, tv, tc))
    
if __name__ == '__main__':
    args = get_args()
    
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)


    logging.info(vars(args))
    train_loader, val_loader, test_loader = get_loader(args)

    model = get_model(args, logging)

    test_model(args, model, test_loader, logging)

    
