import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import time
from rational import *  # assumes rational.py is in the same folder or in PYTHONPATH
import os
from rational_baryrat import get_network

parser = argparse.ArgumentParser(description="BaryratNN")
parser.add_argument("--interval", type=str, default="-10,10")
parser.add_argument("--dim", type=int, default=1)
parser.add_argument("--sample_num", type=int, default=1000)
parser.add_argument("--gap", type=int, default=40)
parser.add_argument("--network", type=str, default='rational')

args = parser.parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Net(torch.nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(1, 50)
        self.R1 = model
        self.ln1 = torch.nn.LayerNorm(50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.R2 = model
        self.ln2 = torch.nn.LayerNorm(50)
        self.fc3 = torch.nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)
        x = self.ln1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.ln2(x)
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible


    class training_points(object):
        def __init__(self, dim, sample_num, interval=args.interval):
            self.dim = dim
            self.sample_num = sample_num
            self.points = torch.ones((sample_num, dim))
            interval_list = list(map(float, interval.split(',')))
            assert len(interval_list) == 2, "Interval should be in format 'min,max'"
            self.interval = torch.tensor(interval_list).view(2, 1).expand(2, dim)  # shape: [2, dim]

        def sample(self):
            for i in range(self.dim):
                self.points[:, i] = torch.linspace(self.interval[0, i], self.interval[1, i], self.sample_num)
            return self.points


    generator = training_points(dim=args.dim, sample_num=args.sample_num, interval=args.interval)
    x = generator.sample()

    #x = torch.unsqueeze(torch.linspace(-10, 10, 1000), dim=1)
    y = torch.sin(2 * x) + 0.1 * torch.rand(x.size())

    x, y = Variable(x), Variable(y)

    ### generate x_support
    gap = args.gap
    x_support = x[::gap, :]
    print("x_support shape:", x_support.shape)

    # Convert interval string to tensor
    # interval = args.interval.split(',')
    # interval_list = list(map(float, interval))
    # interval_matrix = torch.tensor(interval_list).reshape(2, -1)  # shape: [2, dim]
    #
    # # Compute the differences and get the min index
    # differences = torch.abs(interval_matrix[0, :] - interval_matrix[1, :])
    # min_index = torch.argmin(differences).item()
    #
    # lowb, upb = interval_matrix[:, min_index]
    # lowb, upb = min(lowb.item(), upb.item()), max(lowb.item(), upb.item())
    # print("lowb, upb", lowb, upb)
    #
    # # Adjust x_support
    # x_support = x_support[:-1, :] + 0.5 * (upb - lowb) / args.sample_num
    degree = x_support.shape[0]

    mask = ~torch.isin(x, x_support)
    x_train = x[mask]  # Remove x_support from x
    y_train = y[mask]

    #net = Net(UseRational)
    model = get_network(args, degree, x_support)
    net = Net(model)

    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)

    loss_func = torch.nn.MSELoss()

    BATCH_SIZE = 64
    EPOCH = 200

    torch_dataset = Data.TensorDataset(x_train, y_train)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,  # works on Windows now due to the guard
    )

    T = []
    losses = []
    epochs = []
    for epoch in range(EPOCH):
        t1 = time.time()
        for step, (batch_x, batch_y) in enumerate(loader):
            b_x = Variable(batch_x.view(-1,1))
            b_y = Variable(batch_y.view(-1,1))

            prediction = net(b_x)

            loss = loss_func(prediction, b_y)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
            print(f"Step {step}, Gradient Norm: {grad_norm:.4f}")
            optimizer.step()

        t2 = time.time()
        T.append(t2 - t1)
        prediction = net(x)
        loss = loss_func(prediction, y)
        epoch_loss = loss.item()
        losses.append(epoch_loss)
        epochs.append(epoch)
        print("epoch = %d / %d: loss = %f" % (epoch, EPOCH, loss.item()))

    # Save
    epochs_tensor = torch.tensor(epochs).unsqueeze(1)  # shape: [EPOCH, 1]
    losses_tensor = torch.tensor(losses).unsqueeze(1)  # shape: [EPOCH, 1]
    result_tensor = torch.cat((epochs_tensor, losses_tensor), dim=1)  # shape: [EPOCH, 2]
    df = pd.DataFrame(result_tensor.numpy(), columns=['Epoch', 'Loss'])
    df.to_csv(f"loss_by_epoch_{args.network}.csv", index=False)

    # Plotting
    avg_time = sum(T) / len(T)
    print("Avg time per epoch:", avg_time)
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



