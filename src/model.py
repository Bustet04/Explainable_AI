"""
Autoencoder model definition for sleep phase detection.

Architecture:
    Encoder: 24 → 16 → 8 (latent)
    Decoder: 8 → 16 → 24 (reconstruction)
"""

import torch
import torch.nn as nn


class SleepAutoencoder(nn.Module):
    """
    Autoencoder for learning compressed representations of sleep features.
    
    The model compresses 24 spectral/statistical features into an 8-dimensional
    latent space that captures the most important sleep-related patterns.
    """
    
    def __init__(self, input_dim: int = 24, latent_dim: int = 8):
        """
        Initialize autoencoder.
        
        Args:
            input_dim: Number of input features (default: 24)
            latent_dim: Dimension of latent space (default: 8)
        """
        super(SleepAutoencoder, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: 24 → 16 → 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            
            nn.Linear(16, latent_dim),
            nn.ReLU(),
            nn.BatchNorm1d(latent_dim)
        )
        
        # Decoder: 8 → 16 → 24
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Dropout(0.2),
            
            nn.Linear(16, input_dim)
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Latent codes of shape (batch_size, latent_dim)
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.
        
        Args:
            z: Latent tensor of shape (batch_size, latent_dim)
            
        Returns:
            Reconstructed input of shape (batch_size, input_dim)
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: encode then decode.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction, latent_codes)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_latent_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation without gradients (for inference).
        
        Args:
            x: Input tensor
            
        Returns:
            Latent codes
        """
        self.eval()
        with torch.no_grad():
            z = self.encode(x)
        return z


def create_autoencoder(input_dim: int = 24, latent_dim: int = 8,
                      device: str = 'cpu') -> SleepAutoencoder:
    """
    Factory function to create and initialize autoencoder.
    
    Args:
        input_dim: Number of input features
        latent_dim: Dimension of latent space
        device: Device to place model on ('cpu' or 'cuda')
        
    Returns:
        Initialized autoencoder model
    """
    model = SleepAutoencoder(input_dim, latent_dim)
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    
    return model


if __name__ == "__main__":
    # Test model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_autoencoder(device=device)
    
    print("="*70)
    print("AUTOENCODER ARCHITECTURE")
    print("="*70)
    print(model)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Device: {device}")
    
    # Test forward pass
    batch_size = 4
    x_test = torch.randn(batch_size, 24).to(device)
    x_recon, z = model(x_test)
    
    print(f"\nTest forward pass:")
    print(f"  Input shape: {x_test.shape}")
    print(f"  Latent shape: {z.shape}")
    print(f"  Reconstruction shape: {x_recon.shape}")
    print(f"  Compression ratio: {24/8:.1f}x")
