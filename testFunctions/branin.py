import math
import torch
from torch import Tensor
from botorch.test_functions.synthetic import SyntheticTestFunction 
from botorch.utils.sampling import draw_sobol_samples

class BraninTest(SyntheticTestFunction):
    """Branin test function.

    Two-dimensional function (usually evaluated on `[-5, 10] x [0, 15]`):

        B(x) = (x_2 - b x_1^2 + c x_1 - r)^2 + 10 (1-t) cos(x_1) + 10

    Here `b`, `c`, `r` and `t` are constants where `b = 5.1 / (4 * math.pi ** 2)`
    `c = 5 / math.pi`, `r = 6`, `t = 1 / (8 * math.pi)`
    B has 3 minimizers for its global minimum at `z_1 = (-pi, 12.275)`,
    `z_2 = (pi, 2.275)`, `z_3 = (9.42478, 2.475)` with `B(z_i) = 0.397887`.
    """

    dim = 2
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]
    _optimal_value = 0.397887
    _optimizers = [(-math.pi, 12.275), (math.pi, 2.275), (9.42478, 2.475)]

    def __init__(self, config):
        super(BraninTest, self).__init__()
        self.sigma = config['sigma']
        self.repeat_eval = config['repeat_eval']
        self._optimizers = self._optimizers
        self._max_value = self._optimal_value
        self.seed_test = 42
        self.gamma = config['gamma']

    def evaluate_true(self, X: Tensor) -> Tensor:
        t1 = (
            X[..., 1]
            - 5.1 / (4 * math.pi**2) * X[..., 0].pow(2)
            + 5 / math.pi * X[..., 0]
            - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X[..., 0])
        result = t1.pow(2) + t2 + 10
        return torch.Tensor(-result)
    
    def evaluate(self, x: Tensor, seed_eval=None) -> Tensor:
        y_true = self.evaluate_true(x).reshape((-1, 1))
        sigmas = self.get_noise_var(x[:, 0].reshape(-1, 1),
                                    x[:, 1].reshape(-1, 1))

        if seed_eval is not None:
            shape = torch.cat([y_true] * self.repeat_eval, dim=1).shape
            noise = sigmas * torch.randn(shape, generator=torch.Generator().manual_seed(seed_eval))
            y = y_true + noise
        else:
            noise = sigmas * torch.randn_like(torch.cat([y_true] * self.repeat_eval, dim=1))
            y = y_true + noise

        return y
    
    def get_mv(self, x : Tensor) -> Tensor:
        return self.evaluate_true(x).reshape((-1, 1)) - self.gamma * (self.get_noise_var(x[:, 0].reshape(-1, 1),
                                    x[:, 1].reshape(-1, 1))**2)
    
    def get_optimal_mv(self) -> Tensor:
        return max([self._max_value - self.gamma * self.get_noise_var(torch.tensor(x[0]), torch.tensor(x[1]))**2 for x in self._optimizers])

    def get_random_initial_points(self, num_points, seed) -> Tensor:

        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).squeeze()
        return x
    
    def get_domain(self):
        return torch.Tensor(self._bounds).T
    
    def get_noise_var(self, x_0, x_1):
        var1 = self._get_noise_var_1d(x_0.clone().detach(), sigma=self.sigma, shift=3.2)
        var2 = self._get_noise_var_1d(x_1.clone().detach(), sigma=self.sigma, shift=0)
        return var1 * var2
    
    def get_dim(self):
        return self.dim

    @staticmethod
    def _get_noise_var_1d(x, sigma:float=3, shift:float=3):
        return torch.sigmoid((x - shift) * 2) * sigma + 1.0
