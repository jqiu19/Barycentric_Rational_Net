import torch



def get_network(args, degree, x_support):
    if args.network == 'rational_baryrat':
        model = Rational_baryrat(degree=degree, x_support=x_support)
    elif args.network == 'rational':
        model = Rational()
    return model


class Rational_baryrat(torch.nn.Module):
    def __init__(self, degree, x_support, epsilon=1e-2):
        super().__init__()  # Initialize base nn.Module class
        self.degree = degree  # Number of support points (i.e., basis terms)
        self.epsilon = epsilon  # Minimum magnitude to avoid coeffs_w near zero
        self.coeffs_w_raw = torch.nn.Parameter(torch.empty(degree))
        # Raw trainable parameters that will be reparameterized to avoid zero

        self.coeffs_phi = torch.nn.Parameter(torch.empty(degree))
        # Multiplicative coefficients (typically act as the "signal" weights)

        self.reset_parameters()  # Initialize weights
        self.x_support = x_support  # Anchor points (fixed) that define rational basis locations

    def reset_parameters(self):
        self.coeffs_w_raw.data = torch.randn(self.degree) * 0.1
        # Initialize raw weights with small values around 0 to ensure stability after reparameterization

        self.coeffs_phi.data = torch.rand(self.degree)
        # Initialize phi coefficients with uniform random values in [0, 1)

    def get_coeffs_w(self):
        # Reparameterize coeffs_w_raw to avoid values near zero but still allow negative and positive values
        # Formula: sign(x) * (ε + |x|), so it's always outside (-ε, ε) and preserves sign
        return torch.sign(self.coeffs_w_raw) * (self.epsilon + torch.abs(self.coeffs_w_raw))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        coeffs_w = self.get_coeffs_w()  # Apply reparameterization to obtain usable coeffs_w
        coeffs_num = coeffs_w * self.coeffs_phi  # Numerator coefficients (element-wise)
        coeffs_den = coeffs_w  # Denominator uses coeffs_w directly
        coeffs = torch.stack((coeffs_num, coeffs_den), dim=0)
        # Stack into a [2, degree] tensor where 0th row = numerator, 1st row = denominator

        input_dim1 = input.shape[1]  # Input feature dimension (e.g., 1 for scalar input)
        input_dim2 = self.x_support.shape[1]  # x_support feature dimension (should match input)
        assert input_dim1 % input_dim2 == 0  # Ensure dimensions are compatible for broadcasting
        repeat_factor = input_dim1 // input_dim2  # How many times to repeat support to match input

        x_expand = input.unsqueeze(1)  # Shape: [batch_size, 1, input_dim]
        x_support_expand = self.x_support.unsqueeze(0).repeat(1, 1, repeat_factor)
        # Shape: [1, degree, input_dim], repeated to match input batch

        X = x_expand - x_support_expand + 1  # Difference between input and support points
        X = 1.0 / X  # Invert to compute rational basis terms (1 / (x - x_i))

        PQ = torch.einsum('ed,bdw->ebw', coeffs, X)
        # Einstein summation to compute weighted sum of basis terms
        # Output: [2, batch_size, input_dim]

        output = torch.div(PQ[0, ...], PQ[1, ...])
        # Compute element-wise division (numerator / denominator), shape: [batch_size, input_dim]
        return output




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