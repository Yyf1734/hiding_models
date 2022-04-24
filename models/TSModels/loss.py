import torch
import torch.nn.functional as F


class BCELoss(torch.nn.Module):
    def __init__(self, class_num = 8):
        super(BCELoss, self).__init__()
        
        self.class_num = class_num
    
    def forward(self, pred, label, add_softmax=True, reduction='sum'):
        if add_softmax:
            p = F.softmax(pred, dim=-1)
        else:
            p = pred
            
        one_hot = F.one_hot(label, self.class_num)
        p = torch.clamp(p, 1e-6, 1-1e-6)
        
        q = one_hot * torch.log(p) + (1 - one_hot) * torch.log(1 - p)
        loss = -q.sum(dim=-1)
        
        if reduction == 'sum':
            loss = loss.sum(dim=0)
                
        return loss

def calc_bce_loss(pred, label, class_num, add_softmax=True, reduction='sum'):

    # print(pred.size())
    # print(y)
    if add_softmax:
        p = F.softmax(pred, dim=-1)
    else:
        p = pred

    one_hot = F.one_hot(label, class_num)
    p = torch.clamp(p, 1e-6, 1 - 1e-6)

    q = one_hot * torch.log(p) + (1 - one_hot) * torch.log(1 - p)
    loss = -q.sum(dim=-1)

    if reduction == 'sum':
        loss = loss.sum(dim=0)

    return loss