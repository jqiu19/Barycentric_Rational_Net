import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
from rational import *  # assumes rational.py is in the same folder or in PYTHONPATH

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# True for using Rational activation function,
# False for using ReLU
UseRational = True

class Net(torch.nn.Module):
    def __init__(self, UseRational):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 50)
        self.R1 = Rational() if UseRational else F.relu
        self.fc2 = torch.nn.Linear(50, 50)
        self.R2 = Rational() if UseRational else F.relu
        self.fc3 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)
    y = torch.sin(2 * x) + 0.1 * torch.rand(x.size())

    x, y = Variable(x), Variable(y)

    net = Net(UseRational)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    loss_func = torch.nn.MSELoss()

    BATCH_SIZE = 64
    EPOCH = 200

    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # works on Windows now due to the guard
    )

    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x)
            b_y = Variable(batch_y)

            prediction = net(b_x)

            loss = loss_func(prediction, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        prediction = net(x)
        loss = loss_func(prediction, y)
        print("epoch = %d / %d: loss = %f" % (epoch, EPOCH, loss.item()))

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title('Regression Analysis - model 3, Batches', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-11.0, 13.0)
    ax.set_ylim(-1.1, 1.2)
    ax.scatter(x.data.numpy(), y.data.numpy(), color="blue", alpha=0.2)
    prediction = net(x)
    ax.scatter(x.data.numpy(), prediction.data.numpy(), color='green', alpha=0.5)
    plt.savefig('curve_2_model_3_batches.png')
    plt.show()
