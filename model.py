import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, img_shape):
        '''
        input_size - tuple (C,)
        '''
        super().__init__()
        self.img_shape = img_shape
        self.net = nn.Sequential(
            nn.Linear(input_size[0], 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, np.prod(img_shape)),
            nn.Tanh()
        )

    def forward(self, z):
        '''
        Take as input the one-hot vector z of digit
        Return as output a generated vector in shape of image
        '''
        out = self.net(z)
        return out
        
        
class Discriminator(nn.Module):
    
    def __init__(self, inp):
        
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(inp,300),
            nn.ReLU(inplace=True),
            nn.Linear(300,300),
            nn.ReLU(inplace=True),
            nn.Linear(300,200),
            nn.ReLU(inplace=True),
            nn.Linear(200,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.net(x)
        return x
