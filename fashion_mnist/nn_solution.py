import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from torch import nn, optim
from torchvision import datasets, transforms

class Fashion_Classifier(nn.Module):
    def __init__(self, Input):
        super().__init__()

        self.h1 = nn.Linear(Input, 256)
        self.h2 = nn.Linear(256, 128)
        self.h3 = nn.Linear(128, 64)
        self.h4 = nn.Linear(64, 10)

    def forward(self, x):
        # flatten input
        x = x.view(x.shape[0], -1)

        # forward prop
        x = F.relu(self.h1(x))
        x = F.relu(self.h2(x))
        x = F.relu(self.h3(x))
        x = F.log_softmax(self.h4(x), dim=1)

        return x


# transforms to normalize data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

# download minst data
trainset = datasets.FashionMNIST('FashionMNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# set up model and optimiser
model = Fashion_Classifier(784)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)


epochs = 5

for e in range(epochs):
    acc_loss = 0
    for images, labels in trainloader:
        # calculating loss
        logps = model(images)
        loss = criterion(logps, labels)
        acc_loss += loss.item()

        # updating weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch: {}       loss: {}".format(e, acc_loss / len(trainloader)))

print("Done Training")

# get single image
images, labels = next(iter(trainloader))
img = images[0].view(1, 784)

# make prediction on image
with torch.no_grad():
    logits = model.forward(img)

# get probabilities
ps = torch.exp(model(img))

print(ps)
im = Image.fromarray(np.uint8(img.view(28, 28).data.numpy()))
im.show()