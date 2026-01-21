import torch


def get_network(args, degree, x_support, num_units=None):
    """
    num_units: 激活层作用的“通道数/神经元数”，例如 fc1 输出 50 就传 50。
    - 如果 num_units is None：退化到旧行为（每层共享一组 theta，形状 (degree+1,)）
    - 如果 num_units is not None：每个神经元一组 theta，形状 (num_units, degree+1)
    """
    if args.network == 'rational_baryrat':
        model = Rational_baryrat(degree=degree, x_support=x_support, num_units=num_units)
    elif args.network == 'rational':
        model = Rational()
    return model


class Rational_baryrat(torch.nn.Module):
    """
    按 PDF 写法：
        w_j = 1 / Π_{k!=j} (x_j - x_k)
        σ(x) = Σ_j (w_j * θ_j / (x - x_j)) / Σ_j (w_j / (x - x_j))

    关键改动：
    1) w_j 严格跟随 PDF product 公式计算（log-space 稳定版）。
    2) 每个神经元函数的可学习参数 θ_j 不同：theta 参数为 (num_units, degree+1)。
       - input 形状 [B, num_units] 时，逐元素每个通道用自己的 θ。
       - 若 num_units=None，则退化到共享一组 θ（兼容旧用法）。
    3) 假设“每层 50 个神经元用同一组 Chebyshev 节点”，因此 x_support 应传 1D: (degree+1,)。
       若你传的是 (degree+1, dim)，这里会自动取第 0 维列（等价于共享 1D 节点）。
    """
    def __init__(self, degree, x_support, num_units=None, epsilon1=0.01):
        super().__init__()
        self.degree = degree
        self.epsilon1 = epsilon1
        self.num_units = num_units  # None => shared theta, else per-unit theta

        # ---- normalize x_support to 1D (degree+1,) ----
        x_support = torch.as_tensor(x_support)
        if x_support.dim() == 2:
            # 你现在的 x_support 可能是 (degree+1, dim)；按“每个神经元用同一组点”取一列即可
            x_support = x_support[:, 0]
        x_support = x_support.reshape(-1)
        assert x_support.numel() == degree + 1, f"x_support must have {degree+1} nodes, got {x_support.numel()}"

        # 固定节点（不训练）
        self.register_buffer("x_support", x_support)

        # ---- compute barycentric weights w_j by PDF definition ----
        w = self._compute_barycentric_weights_pdf(self.x_support)
        self.register_buffer("coeffs_w", w)  # 固定 w_j（不训练），避免每次 forward 重算

        # ---- learnable theta (coeffs_phi_raw) ----
        if self.num_units is None:
            self.coeffs_phi_raw = torch.nn.Parameter(torch.empty(degree + 1, dtype=x_support.dtype, device=x_support.device))
        else:
            self.coeffs_phi_raw = torch.nn.Parameter(torch.empty(self.num_units, degree + 1, dtype=x_support.dtype, device=x_support.device))

        self.reset_parameters()

    def reset_parameters(self):
        # 每个神经元独立初始化（如果是矩阵，torch.rand 会给每个元素独立随机）
        with torch.no_grad():
            self.coeffs_phi_raw.copy_(torch.rand_like(self.coeffs_phi_raw) * 0.1)

    @staticmethod
    def _compute_barycentric_weights_pdf(x_support_1d: torch.Tensor) -> torch.Tensor:
        """
        w_j = 1 / prod_{k!=j} (x_j - x_k)
        使用 log-abs + sign 来减少 under/overflow 风险。
        """
        x = x_support_1d.reshape(-1)
        n = x.numel()

        # diff[j,k] = x[j] - x[k]
        diff = x.unsqueeze(1) - x.unsqueeze(0)  # (n,n)

        # mask diagonal so it doesn't affect product/sum
        eye = torch.eye(n, dtype=torch.bool, device=x.device)
        diff_no_diag = diff.masked_fill(eye, 1.0)

        # sign and log(abs)
        sign = torch.sign(diff_no_diag).prod(dim=1)           # (n,)
        log_abs = torch.log(torch.abs(diff_no_diag)).sum(dim=1)  # (n,)

        w = sign * torch.exp(-log_abs)  # (n,)
        return w

    def get_coeffs_w(self):
        # 已经预先算好并 register_buffer 了
        return self.coeffs_w

    def get_coeffs_phi(self):
        return self.coeffs_phi_raw

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input:
          - 若 num_units=None: 形状 [B, D] 或任意 (..., D)，共享一套 θ（对所有通道相同）
          - 若 num_units=U:   形状 [B, U] 或任意 (..., U)，每个通道用自己的 θ_u
        返回与 input 同 shape。
        """
        w = self.get_coeffs_w()           # (N,)
        theta = self.get_coeffs_phi()     # (N,) or (U,N)
        xj = self.x_support               # (N,)

        eps = torch.tensor(1e-8, dtype=input.dtype, device=input.device)

        if self.num_units is None:
            # ---- shared theta: theta (N,) ----
            # diff: (..., N)
            diff = input.unsqueeze(-1) - xj
            exact = diff == 0

            diff_safe = torch.where(exact, torch.ones_like(diff), diff)
            inv = 1.0 / diff_safe

            # num/den: (...)
            num = torch.sum(w * theta * inv, dim=-1)
            den = torch.sum(w * inv, dim=-1)
            out = num / (den + 0.0)

            # exact hit -> out = theta[j]
            if exact.any():
                j_idx = exact.to(torch.int64).argmax(dim=-1)  # (...)
                out_exact = theta[j_idx]
                out = torch.where(exact.any(dim=-1), out_exact, out)

            return out

        else:
            # ---- per-unit theta: theta (U, N) ----
            U = self.num_units
            if input.shape[-1] != U:
                raise ValueError(f"Rational_baryrat expected input last dim = {U}, got {input.shape[-1]}")

            # diff: (..., U, N)
            diff = input.unsqueeze(-1) - xj  # broadcast xj to last dim
            exact = diff == 0

            diff_safe = torch.where(exact, torch.ones_like(diff), diff)
            inv = 1.0 / diff_safe

            # reshape w and theta for broadcasting
            # w_:     (..., 1, N)
            # theta_: (..., U, N)
            w_ = w.view(*([1] * (input.dim() - 1)), 1, -1).to(dtype=input.dtype, device=input.device)
            theta_ = theta.view(*([1] * (input.dim() - 1)), U, -1).to(dtype=input.dtype, device=input.device)

            # num/den: (..., U)
            num = torch.sum(w_ * theta_ * inv, dim=-1)
            den = torch.sum(w_ * inv, dim=-1)
            out = num / (den + 0.0)

            # exact hit: out[..., u] = theta[u, j]
            if exact.any():
                j_idx = exact.to(torch.int64).argmax(dim=-1)  # (..., U)
                theta_pick = torch.gather(theta_, dim=-1, index=j_idx.unsqueeze(-1)).squeeze(-1)  # (..., U)
                out = torch.where(exact.any(dim=-1), theta_pick, out)

            return out


class Rational(torch.nn.Module):
    """Rational Activation function.
    It follows:
    `f(x) = P(x) / Q(x),
    where the coefficients of P and Q are initialized to the best rational
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self):
        super().__init__()
        self.coeffs = torch.nn.Parameter(torch.Tensor(4, 2))
        self.reset_parameters()

    def reset_parameters(self):
        self.coeffs.data = torch.Tensor([[1.1915, 0.0],
                                    [1.5957, 2.383],
                                    [0.5, 0.0],
                                    [0.0218, 1.0]])

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self.coeffs.data[0,1].zero_()
        exp = torch.tensor([3., 2., 1., 0.], device=input.device, dtype=input.dtype)
        X = torch.pow(input.unsqueeze(-1), exp)
        PQ = X @ self.coeffs
        output = torch.div(PQ[..., 0], PQ[..., 1])
        return output
