import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channel=1, n_channel=2, kernel_size=2,
                stride=2, dilation=1, padding="valid") -> None:
        super(CNN,self).__init__()

        self.fc_size = 120 * 2
        model = nn.Sequential(
            nn.Conv2d(in_channels=input_channel,out_channels=n_channel,kernel_size=(1,kernel_size),
                      stride=stride,dilation=dilation,padding=padding,bias=False),
            nn.BatchNorm2d(num_features=n_channel),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5),  # 添加Dropout层
        )
        self.model = model
        self.output = nn.Linear(in_features=self.fc_size,out_features=2)

    def forward(self,x):
        x = self.model(x)
        x = x.view(x.size(0),-1)
        output = self.output(x)
        return output
