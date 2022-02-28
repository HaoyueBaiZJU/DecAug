import torch
import torch.nn as nn
from torch.autograd import Variable


def lossc (inputs, target, weight) :
    loss = nn.NLLLoss (reduce = False)
    return loss (inputs, target).view (1, -1).mm (weight).view (1)


def lossb (cfeaturec, weight, cfs) :
    cfeatureb = (cfeaturec.sign () + 1).sign ()
    mfeatureb = 1 - cfeatureb
    loss = Variable (torch.FloatTensor ([0]).cuda ())
    for p in range (cfs) :
        if p == 0 :
            cfeaturer = cfeaturec [:, 1 : cfs]
        elif p == cfs - 1 :
            cfeaturer = cfeaturec [:, 0 : cfs - 1]
        else :
            cfeaturer = torch.cat ((cfeaturec [:, 0 : p], cfeaturec [:, p + 1 : cfs]), 1)
        
        if cfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] != 0 or mfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] != 0 :
            if cfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] == 0 :
                loss += (cfeaturer.t().mm (mfeatureb [:, p : p + 1] * weight) / mfeatureb [:, p : p + 1].t ().mm (weight)).pow (2).sum (0).view (1)
            elif mfeatureb [:, p : p + 1].t ().mm (weight).view (1).data [0] == 0 :
                loss += (cfeaturer.t().mm (cfeatureb [:, p : p + 1] * weight) / cfeatureb [:, p : p + 1].t ().mm (weight)).pow (2).sum (0).view (1)
            else :
                loss += (cfeaturer.t().mm (cfeatureb [:, p : p + 1] * weight) / cfeatureb [:, p : p + 1].t ().mm (weight) -
                         cfeaturer.t().mm (mfeatureb [:, p : p + 1] * weight) / mfeatureb [:, p : p + 1].t ().mm (weight)).pow (2).sum (0).view (1)

    return loss


def lossq (cfeatures, cfs) :
    return - cfeatures.pow (2).sum (1).mean (0).view (1) / cfs

def lossn (cfeatures) :
    return cfeatures.mean (0).pow (2).mean (0).view (1)
