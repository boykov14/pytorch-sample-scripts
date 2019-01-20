import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from torch import nn
from torch import optim
from torchvision import datasets, transforms

class MNIST_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # network architecture
        self.hid1 = nn.Linear(784, 256)
        self.hid2 = nn.Linear(256, 128)
        self.hid3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        # forwards prop of input
        x = F.relu(self.hid1(x))
        x = F.relu(self.hid2(x))
        x = F.relu(self.hid3(x))
        x = F.log_softmax(self.out(x), dim=1)

        return x


# transforms to normalize data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

# download minst data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# get model
model = MNIST_Model()

# define loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 5
for e in range(epochs):

    acc_loss = 0

    for images, labels in trainloader:

        # Flatten images
        images = images.view(images.shape[0], -1)

        # Reset Grads
        optimizer.zero_grad()

        # Forwards Pass
        logps = model(images)

        # Calculate loss
        loss = criterion(logps, labels)
        acc_loss += loss.item()

        # Backwards pass
        loss.backward()

        # Optimizer step
        optimizer.step()

    print("epoch: {}       loss: {}".format(e, acc_loss/len(trainloader)))

print("Done Training")

# get single image
images, labels = next(iter(trainloader))
img = images[0].view(1, 784)

# make prediction on image
with torch.no_grad():
    logits = model.forward(img)

# get probabilities
ps = F.softmax(logits, dim=1)

print(logits)
print(F.softmax(logits))
im = Image.fromarray(np.uint8(img.view(28, 28).data.numpy()))
im.show()

