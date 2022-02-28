import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from model.utils import euclidean_metric, one_hot, count_acc
from model.networks.dropblock import DropBlock
from pytorch_metric_learning import miners, losses

def split_instances(data, num_tasks, num_shot, num_query, num_way, num_class=None):
    num_class = num_way if (num_class is None or num_class < num_way) else num_class

    permuted_ids = torch.zeros(num_tasks, num_shot+num_query, num_way).long()
    for i in range(num_tasks):
        # select class indices
        clsmap = torch.randperm(num_class)[:num_way]
        # ger permuted indices
        for j, clsid in enumerate(clsmap):
            permuted_ids[i, :, j].copy_(
                torch.randperm((num_shot + num_query)) * num_class + clsid
            )

    if torch.cuda.is_available():
        permuted_ids = permuted_ids.cuda()

    support_idx, query_idx = torch.split(permuted_ids, [num_shot, num_query], dim=1)
    return support_idx, query_idx


class SplitBranch(nn.Module):
    ''' Split embedding into multiple subspaces w/ masks '''

    def __init__(self, z_dim, n_concept):
        super().__init__()
        self.z_dim = z_dim
        self.n_concept = n_concept
        self.scale = nn.Linear(z_dim, self.n_concept, bias=False)
        self.bias = nn.Linear(z_dim, self.n_concept, bias=False)
        self.drop = nn.Dropout(0.5)
    
    def mask_proj(self, x, alpha, bias):
        num_task, num_way, num_dim = x.shape
        #print(x.view(1, -1, num_dim).shape)
        #print(alpha.shape)
        #print(alpha.view(self.n_concept, 1, self.z_dim).shape)
        x = torch.mul(x.view(1, -1, num_dim), 1.0 + alpha.view(self.n_concept, 1, self.z_dim)) + bias.view(self.n_concept, 1, self.z_dim)
        return x.view(self.n_concept, num_task, num_way, num_dim)
    
    def forward(self, proto, query=None):
        proto_concepts = self.drop(self.mask_proj(proto, self.scale.weight, self.bias.weight))
        if query is not None:
            query_concepts = self.drop(self.mask_proj(query, self.scale.weight, self.bias.weight))
            return proto_concepts, query_concepts    
        else:
            return proto_concepts
    
class InstanceConcept(nn.Module):
    ''' obtain the instance-concept affiliation '''

    def __init__(self, z_dim, n_concept):
        super().__init__()
        self.z_dim = z_dim
        self.proj = nn.Sequential(nn.Linear(z_dim, z_dim*2), 
                                  nn.ReLU(), 
                                  nn.Linear(z_dim*2, z_dim))
        self.cls = nn.Linear(z_dim, n_concept)
    
    def forward(self, input_data):
        ''' input_data: n_batch x n_way x dim'''
        score = self.cls(self.proj(input_data) + input_data)
        active_index = F.softmax(score, dim=-1)
        return active_index, score
    
class TaskConcept(nn.Module):
    ''' obtain the instance-concept affiliation '''

    def __init__(self, z_dim, n_concept):
        super().__init__()
        self.z_dim = z_dim
        self.proj_inner = nn.Sequential(nn.Linear(z_dim * 3, z_dim*2), 
                                        nn.ReLU(), 
                                        nn.Linear(z_dim*2, z_dim))
        self.proj_outer = nn.Sequential(nn.Linear(z_dim, z_dim*2), 
                                        nn.ReLU(), 
                                        nn.Linear(z_dim*2, z_dim))        
        self.cls = nn.Linear(z_dim, n_concept)
        self.miner = miners.TripletMarginMiner(margin=0.8, type_of_triplets="semihard") # should use large margin if could not mine enough triplets
    
    def forward(self, input_data):
        ''' input_data: n_batch x n_shot x n_way x dim'''
        num_task, num_shot, num_way, num_dim = input_data.shape
        input_data = input_data.view(num_task, -1, num_dim)
        label = torch.arange(num_way, dtype=torch.int16).repeat(num_shot).type(torch.LongTensor)        
        if torch.cuda.is_available():
            label = label.cuda()
        
        # generate triplets
        with torch.no_grad():
            hard_pairs_set = [torch.stack(self.miner(input_data[e], label)).long() for e in range(num_task)]
        
        triplets_emb = [input_data[e][hard_pairs_set[e].flatten()].view(*(hard_pairs_set[e].shape + (num_dim,))) for e in range(num_task)]
        triplets_emb = [triplet_data.permute([1,0,2]).reshape(-1, 3 * num_dim) for triplet_data in triplets_emb]
        triplets_emb = [torch.sum(self.proj_inner(triplet_data), 0) for triplet_data in triplets_emb]
        triplets_emb = torch.stack(triplets_emb)
        score = self.cls(self.proj_outer(triplets_emb))
        active_index = F.softmax(score, dim=-1)
        return active_index

 
def batched_index_select(input, dim, index):
    for ii in range(1, len(input.shape)):
        if ii != dim:
            index = index.unsqueeze(ii)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.expand(expanse)
    return torch.gather(input, dim, index)
    
class MMap(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet(dropblock_size=args.dropblock_size)            
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')
        
        if hasattr(args, 'n_concept'):
            self.n_concept = args.n_concept
        else:
            self.n_concept = args.num_class
            
        self.decompose = SplitBranch(z_dim = hdim, n_concept = self.n_concept)
        self.get_affiliation = InstanceConcept(z_dim = hdim, n_concept = self.n_concept)      
        self.get_task_affiliation = TaskConcept(z_dim= hdim, n_concept = self.n_concept)
    
    def split_instances(self, data):
        args = self.args
        if self.training:
            return  split_instances(data, args.num_tasks, args.shot,
                                    args.query, args.way, args.way)
        else:
            return  (torch.Tensor(np.arange(args.way*args.eval_shot)).long().view(1, args.eval_shot, args.way), 
                     torch.Tensor(np.arange(args.way*args.eval_shot, args.way * (args.eval_shot + args.query))).long().view(1, args.query, args.way))                
    
    
    def forward(self, x, selected_concepts = None, x_aug = None):
        x = x.squeeze(0)
        #print(x.shape)
        support_idx, query_idx = self.split_instances(x)
        #print(support_idx.shape)
        #print(query_idx.shape)
        instance_embs = self.encoder(x)
        #print(instance_embs.shape)
        emb_dim = instance_embs.size(-1) 
        #print(emb_dim)
        num_query = np.prod(query_idx.shape[-2:])
        #print(num_query)
        num_batch = support_idx.shape[0]  
        #print(num_batch)
        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        #print(instance_embs[support_idx.flatten()].shape)
        #print(support.shape)
        query   = instance_embs[query_idx.flatten()].view(num_batch, -1, emb_dim)
        #print(instance_embs[query_idx.flatten()].shape)
        #print(query.shape)

        if self.args.shot == 1 or self.args.eval_shot == 1:
            instance_embs_aug = self.encoder(x_aug)
            support_aug = instance_embs_aug[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
            # extract support set and combine them together
            support = torch.cat([support, support_aug], 1)
            #print(support.shape)

        # get embeddings in all concepts
        proto = support.mean(dim=1) # Ntask x way x dim
        num_proto = proto.shape[1]        
        proto_concepts, query_concepts = self.decompose(proto, query)
        #print(proto_concepts.shape)
        #print(query_concepts.shape)
        
        # proto_summary = self.get_concept(proto_concepts)
        proto_prob, proto_score = self.get_affiliation(proto)
        query_prob, query_score = self.get_affiliation(query)
        # get task-specific affiliation
        task_prob = self.get_task_affiliation(support)
        #print(support.shape)
        #print(task_prob.shape)


        # compute concept-wise logits
        proto_concepts = F.normalize(proto_concepts, p=2, dim=-1)
        logits = torch.bmm(query_concepts.view(-1, num_query, emb_dim), 
                           proto_concepts.view(-1, num_proto, emb_dim).permute([0, 2, 1])).view(self.n_concept, num_batch, num_query, num_proto)    
        # compute concept affiliation probability
        prob = torch.mul(query_prob.unsqueeze(2), proto_prob.unsqueeze(1)).permute([3, 0, 1, 2])
        #print(query_prob.unsqueeze(2).shape)
        #print(proto_prob.unsqueeze(1).shape)
        #print(prob.shape)
        # select the logtis with attentions
        aggregated_logits = torch.sum(torch.mul(torch.mul(logits, prob), task_prob.t().unsqueeze(-1).unsqueeze(-1)), 0).view(-1, num_proto)
        #print(task_prob.t().shape)
        #print(aggregated_logits.shape)
        if self.training and self.args.balance > 0 and self.n_concept == self.args.num_class:
            proto_label = selected_concepts[support_idx[:,[0],:]]
            #print(proto_label.shape)
            query_label = selected_concepts[query_idx]
            #print(query_label.shape)
            # concept loss for binary classification
            num_query_set = int(num_query / num_proto)
            #print(num_query_set)
            # compute the regularizer
            reg_loss = F.cross_entropy(torch.cat([proto_score, query_score], 1).view(-1, self.n_concept),  
                                       torch.cat([proto_label, query_label], 1).view(-1))
            #print(reg_loss.shape)
            return aggregated_logits, reg_loss 
        else:
            return aggregated_logits, None
