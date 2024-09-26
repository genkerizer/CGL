import torch
from torch import nn
import torch.nn.functional as F


class CrossEntropyFusionLoss(nn.Module):
    
    def __init__(self, **kwargs):
        super().__init__()
        self.binary_func = nn.BCEWithLogitsLoss()
        self.entropy_func = nn.CrossEntropyLoss()
        
        
    def forward(self, predicts, targets):
        labels = targets
        logits, cosin_similarity = predicts
        
        logits = logits.flatten()
        
        labels = labels.to(torch.float32)
        logit_loss = self.binary_func(logits, labels)
        
        ground_truth = torch.arange(len(labels),dtype=torch.long, device='cuda')
        constrastive_loss = self.entropy_func(cosin_similarity, ground_truth)
        
        return logit_loss + constrastive_loss
    


    