import torch
import torch.nn as nn

class Inception_model(nn.Module):
    def __init__(self,num_classes,in_channels=3):
        self.num_classes = num_classes
        super(Inception_model,self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.relu = nn.ReLU()
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=192,kernel_size=3,stride=1,padding=1)
        self.relu = nn.ReLU()
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        #parameters will be in the order of in_channels,filters_1x1,filters_3x3_reduce,
        # filters_3x3,filters_5x5_reduce,filters_5x5,filters_pool_proj
        self.inception_3a = Inception_block(192,64,96,128,16,32,32)
        self.inception_3b = Inception_block(256,128,128,192,32,96,64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inception_4a = Inception_block(480,192,96,208,16,48,64)
        self.inception_4b = Inception_block(512,160,112,224,24,64,64)
        self.inception_4c = Inception_block(512,128,128,256,24,64,64)
        self.inception_4d = Inception_block(512,112,144,288,32,64,64)
        self.inception_4e = Inception_block(528,256,160,320,32,128,128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.inception_5a = Inception_block(832,256,160,320,32,128,128)
        self.inception_5b = Inception_block(832,384,192,384,48,128,128)
        self.avg_pool_1 = nn.AvgPool2d(kernel_size=7,stride=1)
        self.flatten = nn.Flatten()
        self.drop_out_1 = nn.Dropout2d(p=0.4)        
        self.linear_1 = nn.Linear(in_features=1024,out_features=1000)
        self.softmax_1 = nn.Softmax(dim=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool_2(x)
        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxpool_3(x)
        x = self.inception_4a(x)
        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)
        x = self.inception_4e(x)
        x = self.maxpool_4(x)
        x = self.inception_5a(x)
        x = self.inception_5b(x)
        x = self.avg_pool_1(x)
        x = self.flatten(x)
        #x = x.reshape(x.shape[0],-1)
        x = self.drop_out_1(x)
        x = self.linear_1(x)
        x = self.softmax_1(x)
        return x

class Inception_block(nn.Module):
    def __init__(self,in_channels,filters_1x1,filters_3x3_reduce,filters_3x3,filters_5x5_reduce,filters_5x5,filters_pool_proj):
        super(Inception_block,self).__init__()

        # 1x1 convolution layer
        self.branch_1 = conv_block(in_channels=in_channels,out_channels=filters_1x1,kernel_size = 1)
        
        # 3x3 convolution layer
        self.branch_2 = nn.Sequential(
            conv_block(in_channels=in_channels,out_channels=filters_3x3_reduce,kernel_size=1),
            conv_block(in_channels=filters_3x3_reduce,out_channels=filters_3x3,kernel_size=3,padding=1)
        )

        # 5x5 convolution layer
        self.branch_3 = nn.Sequential(
            conv_block(in_channels=in_channels,out_channels=filters_5x5_reduce,kernel_size=1),
            conv_block(in_channels=filters_5x5_reduce,out_channels=filters_5x5,kernel_size=5,padding=2)
        )

        # Max pooling layer
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            conv_block(in_channels=in_channels,out_channels=filters_pool_proj,kernel_size=1)
        )
    def forward(self,x):
        return torch.cat([self.branch_1(x),self.branch_2(x),self.branch_3(x),self.branch_4(x)],1)

class conv_block(nn.Module):
    def __init__(self,in_channels,out_channels, **kwargs):
        super(conv_block,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels, **kwargs)
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.conv(x))

if __name__ == '__main__':
    model_inception = Inception_model(1000,3)
    x = torch.randn([3,3,224,224])
    output = model_inception(x)
    print(output.shape)