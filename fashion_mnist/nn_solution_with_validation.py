import numpy as np
from PIL import Image
import time

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

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # flatten input
        x = x.view(x.shape[0], -1)

        # forward prop
        x = self.dropout(F.relu(self.h1(x)))
        x = self.dropout(F.relu(self.h2(x)))
        x = self.dropout(F.relu(self.h3(x)))
        x = F.log_softmax(self.h4(x), dim=1)

        return x


# transforms to augment + normalize data
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                    ])

test_transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                                ])

# download minst data for training
trainset = datasets.FashionMNIST('FashionMNIST_data/', download=True, train=True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# download minst data for testing
testset = datasets.FashionMNIST('FashionMNIST_data/', download=True, train=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# set up model and optimiser
model = Fashion_Classifier(784)
state_dict = torch.load('C:\\Users\\Anton\\Documents\\GitHub\\pytorch-sample-scripts\\fashion_mnist\\weights\\fashion_mnistcheckpoint.pth')
model.load_state_dict(state_dict)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

#setting up gpu
gpu = False
if gpu:
    model = model.cuda()

epochs = 5
steps = 0

train_losses, test_losses = [], []

start = time.time()
for e in range(epochs):
    acc_loss = 0
    for images, labels in trainloader:
        # setting up gpu
        if gpu:
            images = images.cuda()
            labels = labels.cuda()

        # reset grads
        optimizer.zero_grad()

        # calculating loss
        logps = model(images)
        loss = criterion(logps, labels)
        acc_loss += loss.item()

        # updating weights
        loss.backward()
        optimizer.step()

    train_losses.append(acc_loss / len(trainloader))


    test_loss = 0
    accuracy = 0

    # turn of grads for validation
    with torch.no_grad():
        # set model to eval mode
        model.eval()
        for images, labels in testloader:
            # setting up gpu
            if gpu:
                images = images.cuda()
                labels = labels.cuda()

            # calculating loss
            logps = model(images)
            loss = criterion(logps, labels)
            test_loss += loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        # return model to train mode
        model.train()

    test_losses.append(test_loss / len(testloader))

    if test_losses[-1] == max(test_losses):
        print("new best!")
        torch.save(model.state_dict(), 'C:\\Users\\Anton\\Documents\\GitHub\\pytorch-sample-scripts\\fashion_mnist\\weights\\fashion_mnistcheckpoint.pth')

    print("epoch: {}    loss: {} val: {}    acc: {}".format(e, train_losses[-1], test_losses[-1], accuracy/len(testloader)))

print("Done Training in {}".format((time.time() - start)))

# get single image
images, labels = next(iter(trainloader))
img = images[0].view(1, 784)

# make prediction on image
with torch.no_grad():
    if gpu:
        img = img.cuda()
    logits = model.forward(img)

# get probabilities
ps = torch.exp(model(img))

print(ps)
im = Image.fromarray(np.uint8(img.cpu().view(28, 28).data.numpy()))
im.show()

torch.save(model.state_dict(), 'C:\\Users\\Anton\\Documents\\GitHub\\pytorch-sample-scripts\\fashion_mnist\\weights\\fashion_mnistcheckpoint.pth')