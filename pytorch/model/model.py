import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class NN_1_Model(BaseModel):
    
    def __init__(self, input_channels=5, window_size=2048, num_classes=1):
        '''
        Initializes a 1D CNN + LSTM model for time series classification.
        :param input_channels: Number of input channels (e.g. signal types), default is 5 [int]
        :param window_size: Length of the input time window (not used directly, for reference only) [int]
        :param num_classes: Number of output classes for classification [int]
        :return: Initialized NN_1_Model instance [NN_1_Model]
        '''
        super().__init__() # BaseModel constructor
        conv_channels_1=16
        self.conv1 = nn.Conv1d(input_channels, conv_channels_1, kernel_size=32, padding=14)
        next=conv_channels_1
        '''
        conv_channels_2=32
        self.conv2 = nn.Conv1d(conv_channels_1, conv_channels_2, kernel_size=16, padding=7)
        next=conv_channels_2
        '''
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        hidden_dim = 128
        self.lstm = nn.LSTM(input_size=next, hidden_size=hidden_dim, batch_first=True)
        
        self.fc = nn.Linear(hidden_dim,
                            num_classes)
        
    def forward(self, x):  # x: [B, C, T] = Batch size, Channels, Time
        '''
        Defines the forward pass of the model.
        :param x: Input tensor of shape [batch_size, input_channels, window_size],
                  where input_channels is the number of signals (e.g. Bal, S0–S3) [torch.Tensor]
        :return: Output tensor of log-probabilities for each class, shape [batch_size, num_classes]
        :rtype: torch.Tensor
        '''
        x = self.conv1(x)
        x = F.relu(x)
        
        '''
        x = self.conv2(x)
        x = F.relu(x)
        '''
        
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1] 
        x = self.fc(x)
        return x


class NN_2_Model(BaseModel):
    def __init__(self, input_channels=5, window_size=2048, num_classes=6):
        '''
        Initializes a 1D CNN + LSTM model for time series classification.
        :param input_channels: Number of input channels (e.g. signal types), default is 5 [int]
        :param window_size: Length of the input time window (not used directly, for reference only) [int]
        :param hidden_dim: Number of hidden units in the LSTM layer [int]
        :param num_classes: Number of output classes for classification [int]
        :return: Initialized NN_1_Model instance [NN_1_Model]
        '''
        super().__init__() # BaseModel constructor


        conv_channels_1=16
        self.conv1 = nn.Conv1d(input_channels, conv_channels_1, kernel_size=64, padding=31)
        self.bn1 = nn.BatchNorm1d(conv_channels_1)

        conv_channels_2=64
        self.conv2 = nn.Conv1d(conv_channels_1, conv_channels_2, kernel_size=16, padding=5)
        self.bn2 = nn.BatchNorm1d(conv_channels_2)
        out_channel=conv_channels_2

        ''''''
        conv_channels_3=128
        self.conv3 = nn.Conv1d(conv_channels_2, conv_channels_3, kernel_size=8, padding=3)
        self.bn3 = nn.BatchNorm1d(conv_channels_3)
        out_channel = conv_channels_3
        

        self.pool = nn.MaxPool1d(kernel_size=2)

        hidden_dim=64
        bidir = False
        self.lstm = nn.LSTM(input_size=out_channel, hidden_size=hidden_dim, batch_first=True, bidirectional=bidir)

        # Output layer: Fully connected layer for classification [B, hidden_dim] → [B, num_classes]
        self.fc = nn.Linear(hidden_dim * (1+int(bidir)), num_classes)

    def forward(self, x):  # x: [B, C, T] = Batch size, Channels, Time
        '''
        Defines the forward pass of the model.

        :param x: Input tensor of shape [batch_size, input_channels, window_size],
                  where input_channels is the number of signals (e.g. Bal, S0–S3) [torch.Tensor]

        :return: Output tensor of log-probabilities for each class, shape [batch_size, num_classes]
        :rtype: torch.Tensor
        '''
        # [B, 5, T]
        x = self.conv1(x)   # Apply 1D convolution → [B, 16, T]
        x = self.bn1(x)
        x = F.relu(x)       # relu activation function

        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        ''''''
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        

        x = self.pool(x)    # Apply max pooling → [B, 16, T/2]

        x = x.permute(0, 2, 1) # Reshape for LSTM: [B, T/2, 16]

        
        _, (h_n, _) = self.lstm(x) # LSTM output: h_n → [1, B, hidden_dim]

        if self.lstm.bidirectional:
            x = torch.cat((h_n[0], h_n[1]), dim=1)
        else:
            x = h_n[-1] # Final hidden state of LSTM

        x = self.fc(x) # Fully connected [B, num_classes] = [B, 6]


        return x

class AE_Model(nn.Module):
    
    def __init__(self, input_channels=4):
        super().__init__()
        CNN_1_ch = 16
        CNN_2_ch = 32
        CNN_3_ch = 64
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, CNN_1_ch, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(CNN_1_ch, CNN_2_ch, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(CNN_2_ch, CNN_3_ch, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(CNN_3_ch, CNN_2_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(CNN_2_ch),
            nn.ReLU(),
            nn.ConvTranspose1d(CNN_2_ch, CNN_1_ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(CNN_1_ch),
            nn.ReLU(),
            nn.ConvTranspose1d(CNN_1_ch, input_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

