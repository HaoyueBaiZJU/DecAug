import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.utils import euclidean_metric, one_hot, count_acc

from torch.autograd import Variable

import os.path as osp

import torchvision.models as models



class bgor2(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args     
        if args.backbone_class == 'resnet18':

            hdim = 512
            resnet18 = models.resnet18(pretrained=False)
            backbone = resnet18
    
            if args.pretrain:
                backbone.load_state_dict(torch.load(args.init_path), strict=False)
            hdim_i = backbone.fc.in_features
            self.encoder = backbone
            self.encoder.fc = nn.Linear(hdim_i, hdim)
         
        else:
            raise ValueError('')


        self.softmax = nn.Softmax(dim=1)

        self.category_branch = nn.Linear(hdim, hdim)
        self.concept_branch = nn.Linear(hdim, hdim)

        self.relu = nn.ReLU(inplace=True)

        self.fc0 = nn.Linear(hdim, args.num_class)
        self.fcc0 = nn.Linear(hdim, args.num_concept)

        self.classification = nn.Linear(hdim * 2, args.num_class)

    def forward(self, x):

        x = x.squeeze(0)
        B, _, _, _ = x.shape
        
        instance_embs = self.encoder(x)
        instance_embs = torch.reshape(instance_embs, (B, -1))

        category_embs = self.category_branch(instance_embs)

        logits_category = self.fc0(category_embs)
        logits_category = self.softmax(logits_category)


        concept_embs = self.concept_branch(instance_embs)

        logits_concept = self.fcc0(concept_embs)
        logits_concept = self.softmax(logits_concept)

        fused_embs = torch.cat((category_embs, concept_embs), 1)
        
        logits = self.classification(fused_embs)
        logits = self.softmax(logits)


        return logits, logits_category, logits_concept, fused_embs, category_embs, concept_embs

