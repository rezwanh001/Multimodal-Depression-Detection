
import torch 
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = torch.where(targets == 1, 1 - probs, probs)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        focal_loss = BCE_loss * focal_weight
        return torch.sum(focal_loss) / inputs.size(0)
    
##----------------------------------------------------------------------------
import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, lambda_reg=1e-4, alpha=1, gamma=2, focal_weight=1.0, l2_weight=1.0, smooth_eps=0.1):
        super(CombinedLoss, self).__init__()
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.l2_weight = l2_weight
        self.weights = [1/20, 1/80]
        self.smooth_eps = smooth_eps
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
        # self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets, model):
        # Compute the binary cross-entropy loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Apply label smoothing
        smooth_targets = targets * (1 - self.smooth_eps) + 0.5 * self.smooth_eps
        bce_loss = self.bce_loss(inputs, smooth_targets)
        
        # Compute the L2 regularization term
        l2_reg = self.l2_weight * self.lambda_reg * sum(param.norm(2) for param in model.parameters())

        # Compute the Focal Loss component
        probs = torch.sigmoid(inputs)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = torch.where(targets == 1, 1 - probs, probs)
        focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
        focal_loss = self.focal_weight * bce_loss * focal_weight
        focal_loss = focal_loss.mean()  # Normalize by the batch size

        # Combine the losses
        combined_loss = bce_loss + focal_loss + l2_reg
        return combined_loss

# Example usage:
# loss_fn = CombinedLoss(lambda_reg=1e-5, focal_weight=0.5, l2_weight=0.5)
# loss = loss_fn(inputs, targets, model)


