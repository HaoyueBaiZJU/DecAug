import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable

class  met (nn.Module) :
    def __init__ (self, cfs, cls, ndf = 64) :
        super (met, self).__init__ ()
        self.cfs = cfs
        self.ndf = ndf
        self.cfeaconvs = nn.Sequential (
            # 64 x 64 x 3
            nn.Conv2d (3, ndf, 5, 1, 2),
            #nn.BatchNorm2d (ndf, momentum = 0.9),
            nn.ReLU (True),
            nn.MaxPool2d (4, 2, 1),
            # 32 x 32 x (ndf * 2)
            nn.Conv2d (ndf, ndf * 2, 5, 1, 2),
            #nn.BatchNorm2d (ndf * 2, momentum = 0.9),
            nn.ReLU (True),
            nn.MaxPool2d (4, 2, 1),
            # 16 x 16 x (ndf x 2)
            nn.Conv2d (ndf * 2, ndf * 4, 5, 1, 2),
            #nn.BatchNorm2d (ndf * 4, momentum = 0.9),
            nn.ReLU (True),
            nn.MaxPool2d (4, 2, 1),
            # 8 x 8 x (ndf x 4)
            nn.Conv2d (ndf * 4, ndf * 8, 5, 1, 2),
            #nn.BatchNorm2d (ndf * 8, momentum = 0.9),
            nn.ReLU (True),
            nn.MaxPool2d (4, 2, 1),
            # 4 x 4 x (ndf x 8)
            nn.Conv2d (ndf * 8, ndf * 16, 5, 1, 2),
            #nn.BatchNorm2d (ndf * 16, momentum = 0.9),
            nn.ReLU (True),
            nn.MaxPool2d (4, 2, 1),
            #nn.Conv2d (ndf * 16, ndf * 32, 5, 1, 2),
            #nn.ReLU (True),
            #nn.MaxPool2d (4, 2, 1),
        )
        self.cfeafucns = nn.Sequential (
            #nn.Dropout (0.5),
            nn.Linear (ndf * 16 * 4, ndf * 8),
            #nn.BatchNorm1d (ndf * 8, momentum = 0.99),
            nn.ReLU (True),
            #nn.Dropout (0.5),
            nn.Linear (ndf * 8, cfs),
            #nn.BatchNorm1d (cfs, momentum = 0.99),
            nn.Tanh (),
        )
        self.classifier = nn.Sequential (
            nn.Linear (cfs, cls),
            nn.LogSoftmax (),
        )

    def forward (self, input) :
        xc = self.cfeaconvs (input).view (-1, self.ndf * 16 * 4)
        xf = self.cfeafucns (xc)
        output = self.classifier (xf)
        return output, xf


