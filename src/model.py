import json
with open('config.json') as f:
    config = json.load(f)
import torch
import torch.nn as nn


class CharacterLevelCNN(nn.Module):
    def __init__(self, args, number_of_classes):
        super(CharacterLevelCNN, self).__init__()
        
        # Input dropout
        self.dropout_input = nn.Dropout1d(args.dropout_input)
        
        # Convolutional layers with maintained dimensionality
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                args.number_of_characters + len(args.extra_characters),
                256,
                kernel_size=7,
                padding=3  # Maintains input length
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)  # Less aggressive pooling
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Intermediate layers with residual connections
        self.conv3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Final layers with adaptive pooling
        self.conv7 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Adaptive pooling handles variable lengths
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, number_of_classes)
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input: (batch, seq_len, n_chars)
        x = self.dropout_input(x)
        x = x.transpose(1, 2)  # (batch, n_chars, seq_len)
        
        # Feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Residual blocks
        residual = x
        x = self.conv3(x) + residual
        residual = x
        x = self.conv4(x) + residual
        residual = x
        x = self.conv5(x) + residual
        residual = x
        x = self.conv6(x) + residual
        
        # Final features
        x = self.conv7(x)
        x = self.conv8(x)
        
        # Pooling and classification
        x = self.adaptive_pool(x)  # (batch, 512, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        x = self.fc(x)
        
        return x
