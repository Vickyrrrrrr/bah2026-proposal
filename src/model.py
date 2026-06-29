import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """Standard residual block with two convolutional layers and ReLU activation."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x):
        return F.relu(x + self.conv(x))

class MultiHeadCrossAttention(nn.Module):
    """
    Fuses feature maps from the optical and SAR streams.
    Query comes from the optical stream, Key and Value from the SAR stream.
    """
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
    def forward(self, opt_feat, sar_feat):
        B, C, H, W = opt_feat.shape
        
        # Project queries, keys, and values
        q = self.q_proj(opt_feat).view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1) # (B, heads, HW, head_dim)
        k = self.k_proj(sar_feat).view(B, self.num_heads, self.head_dim, H * W)                  # (B, heads, head_dim, HW)
        v = self.v_proj(sar_feat).view(B, self.num_heads, self.head_dim, H * W).transpose(-2, -1) # (B, heads, HW, head_dim)
        
        # Calculate attention scores
        attn_scores = torch.matmul(q, k) / (self.head_dim ** 0.5)  # (B, heads, HW, HW)
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Compute fused values
        out = torch.matmul(attn_weights, v)  # (B, heads, HW, head_dim)
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        return opt_feat + self.out_proj(out)

class LISS4ClearNet(nn.Module):
    """
    LISS-4 ClearNet Architecture:
    - Dual-branch encoder: accepts 3-band cloudy optical and 2-band SAR.
    - Multi-Head Cross-Attention: fuses SAR structure into the optical stream.
    - Residual Decoder: outputs the reconstructed, cloud-free 3-band optical scene.
    """
    def __init__(self, num_res_blocks=16, channel_dim=128):
        super().__init__()
        
        # Optical Encoder Branch (Green, Red, NIR)
        self.opt_conv = nn.Sequential(
            nn.Conv2d(3, channel_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            ResNetBlock(channel_dim),
            ResNetBlock(channel_dim)
        )
        
        # SAR Encoder Branch (VV, VH)
        self.sar_conv = nn.Sequential(
            nn.Conv2d(2, channel_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim),
            nn.ReLU(inplace=True),
            ResNetBlock(channel_dim),
            ResNetBlock(channel_dim)
        )
        
        # Cross-Attention Fusion
        self.cross_attention = MultiHeadCrossAttention(channel_dim, num_heads=8)
        
        # Deep Residual Decoder Loop
        self.res_loop = nn.Sequential(
            *[ResNetBlock(channel_dim) for _ in range(num_res_blocks)]
        )
        
        # Reconstruction Output Head
        self.out_head = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channel_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_dim // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Restrict outputs to range [0, 1]
        )

    def forward(self, optical_cloudy, sar):
        # 1. Encode both branches
        opt_encoded = self.opt_conv(optical_cloudy)
        sar_encoded = self.sar_conv(sar)
        
        # 2. Cross-Attention Fusion (SAR features guide reconstruction of optical stream)
        fused = self.cross_attention(opt_encoded, sar_encoded)
        
        # 3. Residual decoding
        features = self.res_loop(fused)
        
        # 4. Final reconstruction head
        reconstructed = self.out_head(features)
        
        return reconstructed

if __name__ == "__main__":
    # Test pass
    model = LISS4ClearNet(num_res_blocks=8, channel_dim=64)
    opt = torch.randn(2, 3, 256, 256)
    sar = torch.randn(2, 2, 256, 256)
    out = model(opt, sar)
    print("Output tensor shape:", out.shape)
    assert out.shape == (2, 3, 256, 256), "Output shape should match input optical shape!"
