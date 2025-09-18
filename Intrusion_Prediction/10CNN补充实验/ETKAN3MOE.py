import torch
import torch.nn as nn
import torchvision.models as models
from MyKan import KAN

class CTKAN(nn.Module):
    def __init__(self, num_classes=10, flrn=5):
        super(CTKAN, self).__init__()
        
        # EfficientNet-B0 backbone
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.feature_dim = self.efficientnet.classifier[1].in_features  # usually 1280
        self.efficientnet.classifier = nn.Identity()

        # Freeze first flrn layers
        children = list(self.efficientnet.features.children())
        for child in children[:flrn]:
            for param in child.parameters():
                param.requires_grad = False

        # LSTM for spatial-feature-as-sequence modeling
        self.lstm = nn.LSTM(input_size=self.feature_dim, hidden_size=64, num_layers=2,
                            batch_first=True, bidirectional=True)

        # KAN branches
        self.kan_lstm = KAN([128, 64, num_classes])          # Input is 64*2 = 128 from BiLSTM
        self.kan_cnn = KAN([self.feature_dim, 64, num_classes])

        # Learnable residual fusion coefficient (0-1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        batch_size = x.size(0)

        # Extract feature map from EfficientNet
        features = self.efficientnet.features(x)  # Shape: (B, C, H, W)
        B, C, H, W = features.shape

        # Global average pooling for CNN branch
        cnn_gap = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(B, C)  # (B, C)

        # Reshape feature map as pseudo-sequence for LSTM
        lstm_input = features.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        lstm_output, _ = self.lstm(lstm_input)  # (B, H*W, hidden*2)
        last_step = lstm_output[:, -1, :]       # (B, 128)

        # Apply KAN
        moe_res = self.kan_lstm(last_step)      # (B, num_classes)
        cnn_res = self.kan_cnn(cnn_gap)         # (B, num_classes)

        # Residual fusion: cnn_res + alpha * (moe_res - cnn_res)
        final_res = cnn_res + self.alpha * (moe_res - cnn_res)

        return final_res
