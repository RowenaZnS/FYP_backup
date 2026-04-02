"""
Age Factor Extractor for MT-MIM (Multi-Task Mutual Information Minimization)

This module extracts age-related factors from mixed features, enabling
explicit separation of age and identity information.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class AgeFactorExtractor(nn.Module):
    """
    Age Factor Extractor φ(·)
    
    Extracts age factor x_age from mixed feature x.
    The identity feature is then computed as residual: x_id = x - x_age
    
    Args:
        input_dim: Input feature dimension (default: 512)
        hidden_dim: Hidden layer dimension (default: 256)
        output_dim: Output feature dimension (default: 512)
        num_layers: Number of hidden layers (default: 2)
        drop_rate: Dropout rate (default: 0.5)
    """
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512, 
                 num_layers=2, drop_rate=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Build extractor network
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
        
        self.extractor = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Extract age factor from mixed feature.
        
        Args:
            x: Mixed feature tensor of shape (B, input_dim)
            
        Returns:
            x_age: Age factor tensor of shape (B, output_dim)
        """
        x_age = self.extractor(x)
        return x_age


class AgeFactorExtractorV2(nn.Module):
    """
    Alternative Age Factor Extractor with residual connections.
    
    This version uses residual connections for better gradient flow
    and more stable training.
    """
    
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=512, drop_rate=0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # First projection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)
        
        # Second layer with residual
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Output projection
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Extract age factor with residual connections.
        
        Args:
            x: Mixed feature tensor of shape (B, input_dim)
            
        Returns:
            x_age: Age factor tensor of shape (B, output_dim)
        """
        # First layer
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second layer with residual
        identity = out
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = self.relu(out)
        out = self.dropout(out)
        
        # Output projection
        x_age = self.fc3(out)
        
        return x_age

