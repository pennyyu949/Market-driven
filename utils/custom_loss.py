import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, over_penalty, under_penalty):
        super(CustomLoss, self).__init__()
        self.over = over_penalty
        self.under = under_penalty
    
    def forward(self, pred, real):
        
        condition_pred = pred >= real
        loss_over = self.over * ((pred - real) ** 2)
        loss_under = self.under * ((pred - real) ** 2)
        loss_power = torch.where(
            condition_pred, loss_over, loss_under
        )
        loss_power = torch.mean(loss_power)
        
        return loss_power
    
class HubeiLoss(nn.Module):
    def __init__(self, over_penalty, under_penalty):
        super(HubeiLoss, self).__init__()
        self.over = over_penalty
        self.under = under_penalty
    
    def forward(self, pred, real):
        diff_real_pred = real - pred
        diff_pred_real = pred - real

        condition_under = (diff_real_pred > 2) | (diff_real_pred > self.under * pred)
        condition_over = (diff_pred_real > 2) | (diff_pred_real > self.over * pred)

        loss_under = self.under * ((real - pred) ** 2)
        loss_over = self.over * ((real - pred) ** 2)
        loss_default = ((real - pred) ** 2)

        loss = torch.where(condition_under, loss_under, torch.where(condition_over, loss_over, loss_default))
        loss_power = torch.mean(loss)
                
        return loss_power