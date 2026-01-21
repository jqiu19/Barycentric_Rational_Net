import torch
import numpy as np
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
from data import get_data

torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="BaryratNN")
parser.add_argument("--interval", type=str, default="-10,10")  # dim=3: "-10,10;-5,5;0,1"
parser.add_argument("--dim", type=int, default=1)
parser.add_argument("--degree", type=int, default=64)
parser.add_argument("--sample_num", type=int, default=1000)
parser.add_argument("--gap", type=int, default=40)
parser.add_argument("--network", type=str, default='rational_baryrat')
parser.add_argument("--output_dim", type=int, default=1)
parser.add_argument("--data", type=str, default='sin_freq')

args = parser.parse_args()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 🔥 Explicitly use CUDA device 7
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"Device name: {torch.cuda.get_device_name(7)}")



class Net(torch.nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(args.dim, 50)
        self.R1 = get_network(args, degree, x_support, num_units=50)
        self.ln1 = torch.nn.LayerNorm(50)
        self.fc2 = torch.nn.Linear(50, 50)
        self.R2 = get_network(args, degree, x_support, num_units=50)
        self.ln2 = torch.nn.LayerNorm(50)
        self.fc3 = torch.nn.Linear(50, args.output_dim)
        self.R3 = get_network(args, degree, x_support, num_units=50)

    def forward(self, x):
        x = self.fc1(x)
        x = self.R1(x)      # expects per-neuron theta if using new Rational_baryrat
        x = self.ln1(x)
        x = self.fc2(x)
        x = self.R2(x)
        x = self.ln2(x)
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(1)  # reproducible

    class training_points(object):
        def __init__(self, dim, sample_num, interval=args.interval, device="cpu", dtype=torch.float64):
            self.dim = dim
            self.sample_num = sample_num

            interval_str = interval.split(';')
            assert len(interval_str) == dim, (
                f"interval must have exactly {dim} segments separated by ';'. "
                f"Got {len(interval_str)}. Example: dim=3 -> '-10,10;-5,5;0,1'"
            )

            # interval: shape (2, dim), interval[0,i]=a_i, interval[1,i]=b_i
            self.interval = torch.tensor(
                [list(map(float, s.split(','))) for s in interval_str],
                dtype=dtype,
                device=device
            ).T

            if not torch.all(self.interval[0] < self.interval[1]):
                raise ValueError(f"Each interval must satisfy a_i < b_i. Got:\n{self.interval}")

        def sample(self):
            points = torch.empty(self.sample_num, self.dim, dtype=self.interval.dtype, device=self.interval.device)
            for i in range(self.dim):
                low, high = self.interval[:, i]
                points[:, i].uniform_(low, high)
            return points

    # NOTE: keep dataset tensors on CPU if num_workers>0
    generator = training_points(dim=args.dim, sample_num=args.sample_num, interval=args.interval, device="cpu", dtype=torch.float64)
    x = generator.sample()                 # CPU
    generate_data = get_data(args.data)

    # IMPORTANT: your get_data may include numpy-only functions; keep x on CPU here
    y = generate_data(x)                   # CPU (either torch or numpy output depending on data.py)

    # ensure y is torch.Tensor (TensorDataset needs tensors)
    if not torch.is_tensor(y):
        y = torch.as_tensor(y, dtype=torch.float64)

    # --- Chebyshev nodes: same for all neurons in a layer (1D, degree+1) ---
    degree = args.degree
    j = torch.arange(degree + 1, dtype=torch.float64, device=device)
    x_support = 10.0 * torch.cos(j * torch.pi / degree)  # (degree+1,)
    x_support = x_support.to(device)

    # --- build model: per-neuron theta => pass num_units=50 ---
    # R1, R2 都作用在 [B,50] 上，所以 num_units=50
    model = get_network(args, degree, x_support, num_units=50)
    net = Net(model).to(device)

    # Optimizer with different learning rates
    param_groups = [
        {'params': [p for name, p in net.named_parameters() if 'coeffs_phi_raw' not in name], 'lr': 1e-3},
        {'params': [p for name, p in net.named_parameters() if 'coeffs_phi_raw' in name], 'lr': 3e-3}
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)
    loss_func = torch.nn.MSELoss()

    BATCH_SIZE = 64
    EPOCH = 200
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH, eta_min=1e-6)

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    # eval tensors on GPU (safe; not used by DataLoader workers)
    x_eval = x.to(device, non_blocking=True)
    y_eval = y.to(device, non_blocking=True)

    T = []
    losses = []
    epochs = []

    for epoch in range(EPOCH):
        t1 = time.time()

        for step, (batch_x, batch_y) in enumerate(loader):
            # move batch to GPU
            b_x = batch_x.to(device, non_blocking=True)
            b_y = batch_y.to(device, non_blocking=True)

            prediction = net(b_x)
            loss = loss_func(prediction, b_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t2 = time.time()
        T.append(t2 - t1)

        with torch.no_grad():
            prediction = net(x_eval)
            epoch_loss = loss_func(prediction, y_eval).item()
        losses.append(epoch_loss)
        epochs.append(epoch)

        total_norm = 0.0
        coeffs_phi_grad_norm = 0.0
        for name, p in net.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                if 'coeffs_phi_raw' in name:
                    coeffs_phi_grad_norm = param_norm.item()
        total_norm = total_norm ** 0.5

        print("epoch = %d / %d: loss = %f, grad_norm = %f, coeffs_phi_grad = %f" %
              (epoch, EPOCH, epoch_loss, total_norm, coeffs_phi_grad_norm))

        scheduler.step()

    # Save results
    epochs_tensor = torch.tensor(epochs).unsqueeze(1)
    losses_tensor = torch.tensor(losses).unsqueeze(1)
    result_tensor = torch.cat((epochs_tensor, losses_tensor), dim=1)
    df = pd.DataFrame(result_tensor.cpu().numpy(), columns=['Epoch', 'Loss'])
    df.to_csv(f"loss_by_epoch_{args.network}.csv", index=False)

    # Plotting
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_title(f'Regression Analysis - Baryrat with degree {args.degree}', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-11.0, 13.0)
    ax.set_ylim(-1.1, 1.2)
    ax.scatter(x.cpu().numpy(), y.cpu().numpy(), color="blue", alpha=0.2)
    with torch.no_grad():
        pred_plot = net(x_eval).cpu().numpy()
    ax.scatter(x.cpu().numpy(), pred_plot, color='green', alpha=0.5)
    plt.savefig('curve_2_model_3_batches.png')
    plt.show()

    print("Avg time per epoch:", sum(T) / len(T))
