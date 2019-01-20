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

# define loss
criterion = nn.NLLLoss()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# get out data
images, labels = next(iter(trainloader))

# Flatten images
images = images.view(images.shape[0], -1)

# Reset Grads
optimizer.zero_grad()

# Forwards Pass
logps = model(images)

# Calculate loss
loss = criterion(logps, labels)

# Backwards pass
loss.backward()

# Optimizer step
optimizer.step()

print(logps.shape)
print(loss)

