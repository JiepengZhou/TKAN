import torch
import torch.nn as nn
from MyKan import KAN

class ViTKAN(nn.Module):
    def __init__(self, image_size=(32, 6), patch_size=(4, 3), in_channels=1, num_classes=10,
                 dim=12, depth=6, heads=4, mlp_dim=256):
        super().__init__()

        # 1. Patch计算
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = in_channels * patch_size[0] * patch_size[1]

        # 确保 patch_dim 和 dim 一致
        assert patch_dim == dim, f"patch_dim ({patch_dim}) must be equal to dim ({dim})"

        self.patch_size = patch_size
        self.num_patches = num_patches

        # 2. Patch embedding
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # 3. Positional encoding & Transformer
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. LSTM + KAN as before
        self.lstm = nn.LSTM(input_size=dim, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.kan_vit = KAN([dim, 64, num_classes])
        self.kan_lstm = KAN([128, 64, num_classes])
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        if x.ndim == 3:  # (B, H, W)
            x = x.unsqueeze(1)  # -> (B, 1, H, W)

        B, C, H, W = x.shape
        ph, pw = self.patch_size

        # Patchify
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, H//ph, W//pw, ph, pw)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, H//ph, W//pw, C, ph, pw)
        x = x.view(B, -1, C * ph * pw)  # (B, num_patches, patch_dim)
        x_embed = self.patch_to_embedding(x)  # (B, num_patches, dim)

        # ViT path
        cls_token = self.cls_token.expand(B, -1, -1)
        vit_input = torch.cat((cls_token, x_embed), dim=1)
        vit_input = vit_input + self.pos_embedding[:, :vit_input.size(1)]
        vit_output = self.transformer(vit_input)
        cls_feature = vit_output[:, 0]
        vit_res = self.kan_vit(cls_feature)

        # LSTM path
        lstm_out, _ = self.lstm(x_embed)
        last_hidden = lstm_out[:, -1, :]
        lstm_res = self.kan_lstm(last_hidden)

        out = vit_res + self.alpha * (lstm_res - vit_res)
        return out
