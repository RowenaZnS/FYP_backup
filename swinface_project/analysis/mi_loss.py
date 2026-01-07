"""
Mutual Information Minimization Loss for MT-MIM

This module provides various loss functions for minimizing mutual information
between identity and age features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILoss(nn.Module):
    """
    Mutual Information Minimization Loss using InfoNCE formulation.
    
    The goal is to minimize I(x_id; x_age) by making the MI estimator
    unable to distinguish between positive pairs (x_id, x_age) and
    negative pairs (x_id, x_age_shuffled).
    
    Args:
        temperature: Temperature for scaling similarities (default: 0.1)
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x_id, x_age, x_age_shuffled):
        """
        Compute MI minimization loss.
        
        Args:
            x_id: Identity feature (B, D)
            x_age: Age feature (B, D) - positive samples
            x_age_shuffled: Shuffled age feature (B, D) - negative samples
            
        Returns:
            mi_loss: Scalar loss value
        """
        # Normalize features
        x_id_norm = F.normalize(x_id, p=2, dim=1)
        x_age_norm = F.normalize(x_age, p=2, dim=1)
        x_age_shuffled_norm = F.normalize(x_age_shuffled, p=2, dim=1)
        
        # Compute similarities
        # Positive pair: (x_id, x_age) - should be low
        pos_sim = torch.sum(x_id_norm * x_age_norm, dim=1) / self.temperature  # (B,)
        
        # Negative pair: (x_id, x_age_shuffled) - should be similar to positive
        neg_sim = torch.sum(x_id_norm * x_age_shuffled_norm, dim=1) / self.temperature  # (B,)
        
        # InfoNCE-style loss
        # We want pos_sim ≈ neg_sim (i.e., x_id doesn't help distinguish x_age)
        # This is equivalent to minimizing MI
        loss = -torch.log(torch.sigmoid(neg_sim - pos_sim) + 1e-8)
        
        return loss.mean()


class MILossContrastive(nn.Module):
    """
    Contrastive MI Loss with margin.
    
    Uses a margin-based contrastive loss to enforce that positive and negative
    pairs have similar similarities.
    
    Args:
        margin: Margin for contrastive loss (default: 0.1)
    """
    
    def __init__(self, margin=0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, x_id, x_age, x_age_shuffled):
        """
        Compute margin-based MI loss.
        
        Args:
            x_id: Identity feature (B, D)
            x_age: Age feature (B, D)
            x_age_shuffled: Shuffled age feature (B, D)
            
        Returns:
            mi_loss: Scalar loss value
        """
        # Compute cosine similarities
        pos_sim = F.cosine_similarity(x_id, x_age, dim=1)
        neg_sim = F.cosine_similarity(x_id, x_age_shuffled, dim=1)
        
        # Margin loss: want pos_sim to be close to neg_sim
        # If pos_sim > neg_sim + margin, there's a penalty
        loss = F.relu(pos_sim - neg_sim + self.margin)
        
        return loss.mean()


class MILossMSE(nn.Module):
    """
    MSE-based MI Loss.
    
    Uses MSE between predicted and actual age features from MI estimator.
    High MSE means MI estimator cannot predict well, indicating low MI.
    
    This loss is used to train the MI estimator (not the main network).
    """
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, x_age_pred, x_age_target):
        """
        Compute MSE loss for MI estimator training.
        
        Args:
            x_age_pred: Predicted age feature from MI estimator (B, D)
            x_age_target: Target age feature (B, D)
            
        Returns:
            mse_loss: Scalar loss value
        """
        return self.mse(x_age_pred, x_age_target)


class MILossNCE(nn.Module):
    """
    NCE (Noise Contrastive Estimation) based MI Loss.
    
    A more sophisticated version that uses multiple negative samples
    for better MI estimation.
    
    Args:
        temperature: Temperature for scaling (default: 0.07)
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, x_id, x_age):
        """
        Compute NCE-based MI loss using batch as negatives.
        
        Args:
            x_id: Identity features (B, D)
            x_age: Age features (B, D)
            
        Returns:
            nce_loss: Scalar loss value (negated for MI minimization)
        """
        batch_size = x_id.size(0)
        
        # Normalize
        x_id = F.normalize(x_id, dim=1)
        x_age = F.normalize(x_age, dim=1)
        
        # Compute all pairwise similarities
        # sim[i, j] = similarity between x_id[i] and x_age[j]
        sim_matrix = torch.mm(x_id, x_age.t()) / self.temperature  # (B, B)
        
        # Positive pairs are on diagonal
        pos_sim = torch.diag(sim_matrix)  # (B,)
        
        # For MI minimization, we want the diagonal (positive pairs) 
        # to be indistinguishable from off-diagonal (negative pairs)
        
        # Standard InfoNCE would maximize pos_sim relative to all others
        # For MI minimization, we want to minimize this distinction
        
        # Compute InfoNCE loss (but we'll negate it for MI minimization)
        # log(exp(pos) / sum(exp(all)))
        log_prob = pos_sim - torch.logsumexp(sim_matrix, dim=1)
        
        # For MI minimization, we want low InfoNCE (high entropy)
        # So we minimize the negative of standard InfoNCE
        # Equivalently, we want log_prob to be low (close to -log(B))
        
        # Target: uniform distribution over all samples
        target_log_prob = -torch.log(torch.tensor(batch_size, dtype=torch.float, device=x_id.device))
        
        # Loss: push log_prob towards uniform
        loss = (log_prob - target_log_prob).pow(2).mean()
        
        return loss


class CombinedMILoss(nn.Module):
    """
    Combined MI Loss using multiple loss components.
    
    Args:
        temperature: Temperature for InfoNCE (default: 0.1)
        margin: Margin for contrastive loss (default: 0.1)
        infonce_weight: Weight for InfoNCE loss (default: 1.0)
        contrastive_weight: Weight for contrastive loss (default: 0.5)
    """
    
    def __init__(self, temperature=0.1, margin=0.1, 
                 infonce_weight=1.0, contrastive_weight=0.5):
        super().__init__()
        
        self.infonce_loss = MILoss(temperature=temperature)
        self.contrastive_loss = MILossContrastive(margin=margin)
        
        self.infonce_weight = infonce_weight
        self.contrastive_weight = contrastive_weight
    
    def forward(self, x_id, x_age, x_age_shuffled):
        """
        Compute combined MI loss.
        
        Args:
            x_id: Identity feature (B, D)
            x_age: Age feature (B, D)
            x_age_shuffled: Shuffled age feature (B, D)
            
        Returns:
            total_loss: Combined loss value
            loss_dict: Dictionary of individual losses
        """
        loss_infonce = self.infonce_loss(x_id, x_age, x_age_shuffled)
        loss_contrastive = self.contrastive_loss(x_id, x_age, x_age_shuffled)
        
        total_loss = (self.infonce_weight * loss_infonce + 
                     self.contrastive_weight * loss_contrastive)
        
        loss_dict = {
            'mi_infonce': loss_infonce.item(),
            'mi_contrastive': loss_contrastive.item(),
            'mi_total': total_loss.item()
        }
        
        return total_loss, loss_dict


def shuffle_batch(x):
    """
    Shuffle a batch of features along the batch dimension.
    
    Args:
        x: Feature tensor of shape (B, D)
        
    Returns:
        x_shuffled: Shuffled feature tensor of shape (B, D)
    """
    batch_size = x.size(0)
    perm = torch.randperm(batch_size, device=x.device)
    return x[perm]


def shuffle_batch_exclude_self(x):
    """
    Shuffle a batch ensuring no element maps to itself.
    
    Args:
        x: Feature tensor of shape (B, D)
        
    Returns:
        x_shuffled: Shuffled feature tensor where x_shuffled[i] != x[i]
    """
    batch_size = x.size(0)
    
    if batch_size <= 1:
        return x
    
    # Create a derangement (permutation with no fixed points)
    perm = torch.arange(batch_size, device=x.device)
    
    # Simple approach: shift by 1
    perm = (perm + 1) % batch_size
    
    return x[perm]

