import torch
import torch.nn as nn
from MyKan import KAN

class ViTKAN(nn.Module):
    def __init__(self, image_size=(32, 6), patch_size=(4, 3), in_channels=3, num_classes=10,
                 dim=128, depth=6, heads=4, mlp_dim=256):
        super().__init__()

        assert image_size[0] % patch_size[0] == 0 and image_size[1] % patch_size[1] == 0
        num_patches = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])
        patch_dim = in_channels * patch_size[0] * patch_size[1]

        self.patch_size = patch_size
        self.num_patches = num_patches

        # Patch embedding
        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # CLS token and pos embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # Ensure this is correctly sized

        # ViT encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # LSTM branch
        self.lstm = nn.LSTM(input_size=dim, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True)

        # KAN branches
        self.kan_vit = KAN([dim, 64, num_classes])
        self.kan_lstm = KAN([128, 64, num_classes])  # 64*2 from BiLSTM

        # Fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        # print(f"[输入] x.shape: {x.shape}")  # (B, 3, 32, 6)
    
        # Patch embedding
        x = x.unfold(2, ph, ph).unfold(3, pw, pw)  # (B, C, H//ph, W//pw, ph, pw)
        # print(f"[unfold后] x.shape: {x.shape}")
    
        x = x.permute(0, 2, 3, 1, 4, 5)  # (B, H//ph, W//pw, C, ph, pw)
        # print(f"[permute后] x.shape: {x.shape}")
    
        x = x.contiguous().view(B, -1, C * ph * pw)  # (B, N_patches, patch_dim)
        # print(f"[reshape后] patch序列 x.shape: {x.shape}")
    
        x_embed = self.patch_to_embedding(x)  # (B, N_patches, dim)
        # print(f"[patch_to_embedding后] x_embed.shape: {x_embed.shape}")
    
        # ViT path
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        # print(f"[cls_token扩展后] cls_token.shape: {cls_token.shape}")
    
        vit_input = torch.cat((cls_token, x_embed), dim=1)  # (B, N_patches+1, dim)
        # print(f"[拼接后] vit_input.shape: {vit_input.shape}")
    
        pos_embedding = self.pos_embedding[:, :vit_input.size(1)]  # 截取合适的 positional embedding
        # print(f"[截取pos_embedding后] pos_embedding.shape: {pos_embedding.shape}")
    
        vit_input = vit_input + pos_embedding
        # print(f"[加入pos_embedding后] vit_input.shape: {vit_input.shape}")
    
        vit_output = self.transformer(vit_input)
        # print(f"[transformer输出] vit_output.shape: {vit_output.shape}")
    
        cls_feature = vit_output[:, 0]
        # print(f"[提取cls token特征] cls_feature.shape: {cls_feature.shape}")
    
        vit_res = self.kan_vit(cls_feature)
        # print(f"[ViT路径结果] vit_res.shape: {vit_res.shape}")
    
        # LSTM path
        lstm_out, _ = self.lstm(x_embed)  # (B, N_patches, 128)
        # print(f"[LSTM输出] lstm_out.shape: {lstm_out.shape}")
    
        last_hidden = lstm_out[:, -1, :]  # (B, 128)
        # print(f"[LSTM最后一步输出] last_hidden.shape: {last_hidden.shape}")
    
        lstm_res = self.kan_lstm(last_hidden)  # (B, num_classes)
        # print(f"[LSTM路径结果] lstm_res.shape: {lstm_res.shape}")
    
        # Fusion
        out = vit_res + self.alpha * (lstm_res - vit_res)
        # print(f"[融合输出] out.shape: {out.shape}")
        return out
