import torch
import torch.nn as nn

class ViT(nn.Module):
    def __init__(self, image_size=(32, 6), patch_size=(4, 3), in_channels=3, num_classes=10, dim=128, depth=6, heads=4, mlp_dim=256):
        super().__init__()

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0, "Image must be divisible by patch size"
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = in_channels * patch_size[0] * patch_size[1]

        self.patch_size = patch_size

        # Patch embedding: flatten and linear projection
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classification head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        assert H == 32 and W == 6, f"Expected input size (32, 6), but got ({H}, {W})"

        # (B, C, H, W) -> (B, patch_num, patch_dim)
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, H//ph, W//pw, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5)            # (B, H//ph, W//pw, C, ph, pw)
        x = x.contiguous().view(B, -1, C * ph * pw)  # (B, N_patches, patch_dim)

        x = self.patch_to_embedding(x)  # (B, N_patches, dim)

        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)          # (B, N_patches + 1, dim)

        x = x + self.pos_embedding[:, :x.size(1)]      # Add positional encoding

        x = self.transformer(x)                        # (B, N_patches + 1, dim)
        cls_output = x[:, 0]                           # Take the cls token

        out = self.mlp_head(cls_output)                # (B, num_classes)
        return out
