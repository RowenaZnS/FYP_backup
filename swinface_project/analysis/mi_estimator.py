"""
Mutual Information Estimator for MT-MIM (Multi-Task Mutual Information Minimization)

This module estimates mutual information between identity and age features,
used to enforce statistical independence through MI minimization.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class MIEstimator(nn.Module):
    """
    Mutual Information Estimator Q_σ
    
    Attempts to predict x_age from x_id. If successful, it means x_id contains
    age information (high MI). The goal is to make this prediction fail,
    indicating low MI and thus feature independence.
    
    Args:
        input_dim: Input feature dimension (default: 512)
        hidden_dim: Hidden layer dimension (default: 256)
        output_dim: Output feature dimension (default: 512)
        num_layers: Number of hidden layers (default: 2)
        drop_rate: Dropout rate (default: 0.3)
    """
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512,
                 num_layers=2, drop_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build estimator network
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(drop_rate))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(drop_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.estimator = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_id):
        """
        Predict age feature from identity feature.
        
        Args:
            x_id: Identity feature tensor of shape (B, input_dim)
            
        Returns:
            x_age_pred: Predicted age feature tensor of shape (B, output_dim)
        """
        x_age_pred = self.estimator(x_id)
        return x_age_pred


class BilinearMIEstimator(nn.Module):
    """
    Bilinear Mutual Information Estimator
    
    Uses bilinear interaction to estimate MI between x_id and x_age.
    This is an alternative approach that directly scores the compatibility
    between identity and age features.
    
    Args:
        feature_dim: Feature dimension (default: 512)
        hidden_dim: Hidden dimension for bilinear interaction (default: 128)
    """
    
    def __init__(self, feature_dim=512, hidden_dim=128):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Project features to lower dimension for bilinear interaction
        self.proj_id = nn.Linear(feature_dim, hidden_dim)
        self.proj_age = nn.Linear(feature_dim, hidden_dim)
        
        # Bilinear weight matrix
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x_id, x_age):
        """
        Compute compatibility score between identity and age features.
        
        Args:
            x_id: Identity feature tensor of shape (B, feature_dim)
            x_age: Age feature tensor of shape (B, feature_dim)
            
        Returns:
            score: Compatibility score of shape (B, 1)
        """
        # Project features
        h_id = self.proj_id(x_id)
        h_age = self.proj_age(x_age)
        
        # Bilinear interaction
        score = self.bilinear(h_id, h_age)
        
        return score

