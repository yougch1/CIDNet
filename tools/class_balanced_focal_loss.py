
import numpy as np
import torch
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class CB_loss(nn.Module):
    def __init__(self,  gamma=2, beta=0.999,samples_per_cls=[3601,926,5345],no_of_classes=3,size_average=True, ignore_index=255,):
        super(CB_loss, self).__init__()
        self.samples_per_cls=samples_per_cls
        self.beta=beta
        self.gamma = gamma
        self.no_of_classes=no_of_classes
        self.ignore_index = ignore_index
        self.size_average = size_average # 对batch里面的数据取均值/求和
    def forward(self, inputs, targets):
    
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes
    
        labels_one_hot = F.one_hot(targets, self.no_of_classes).float()
    
        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)
        #BCLoss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        BCLoss = F.binary_cross_entropy_with_logits(inputs, labels_one_hot, reduction="none")
        
        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * labels_one_hot * inputs - self.gamma * torch.log(1 +
                                                                               torch.exp(-1.0 * inputs)))

        loss = modulator * BCLoss

        weighted_loss = weights * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(targets)
        
        return focal_loss

if __name__ == '__main__':
    # output = torch.tensor([4.0, 5.0, 10.0])
    #
    # label = torch.tensor([2, 1, 1], dtype=torch.long)
    logits = torch.rand(10, 3).float()
    print("logit", logits)
    labels = torch.randint(0, 3, size=(10,))
    print("labels", labels)
    # beta = 0.9999
    # gamma = 2.0
    # samples_per_cls = [2,3,1,2,2]
    
    cb_loss=CB_loss()
    focal_loss = cb_loss(logits, labels)
    print(focal_loss)
