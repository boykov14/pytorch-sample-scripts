from torch import nn

class MNIST_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # network architecture
        self.hid = nn.Linear(784, 256)
        self.out = nn.Linear(256, 10)

        #define activations
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x):
        # forwards prop of input
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        x = self.softmax(x)

        return x

model = MNIST_Model()
print(model)