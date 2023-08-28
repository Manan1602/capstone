import torch.nn as nn
import torchvision
class embed(nn.Module):
        def __init__(self,config):
            super().__init__()
            resnet = torchvision.models.resnet50(weights='IMAGENET1K_V1')
            modules = list(resnet.children())[:-2]
            self.resnet = nn.Sequential(*modules)
            for i in self.resnet.parameters():
                i.requires_grad = False
            self.flat = nn.Flatten(start_dim = 1)
            self.model = nn.Sequential(
                nn.Linear(32768,16000),
                nn.ReLU(True),
                nn.Linear(16000,config.embedding_size)
            )
        def forward(self,x):
            x = self.resnet(x)
            x = self.flat(x)
            x = self.model(x)
            return x