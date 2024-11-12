from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
import torch
from torch import Tensor
from botorch.acquisition.objective import PosteriorTransform
from botorch.utils.transforms import t_batch_mode_transform
from botorch.exceptions import UnsupportedError

class RiskAverseUpperConfidenceBound(AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        model_varproxi: Model,
        beta: float | Tensor,
        beta_varproxi: float | Tensor,
        gamma: float | Tensor,
        posterior_transform: PosteriorTransform | None = None,
        maximize: bool = True,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform)
        self.maximize = maximize
        self.model_varproxi = model_varproxi
        #if beta/varproxi/gamma is a scalar, convert it to a tensor
        if not torch.is_tensor(beta):
            beta = torch.tensor([beta])
        self.register_buffer("beta", beta)
        if not torch.is_tensor(beta_varproxi):
            beta_varproxi = torch.tensor([beta_varproxi])
        self.register_buffer("beta_varproxi", beta_varproxi)
        if not torch.is_tensor(gamma):
            gamma = torch.tensor(gamma)
        self.register_buffer("gamma", gamma)

    def _get_model_mean_and_sigma(self, X: Tensor, model: Model, min_var: float = 1e-12) -> Tensor:
        r"""Get the posterior variance of the model_varproxi at the candidate set X.

        Args:
            X: A `(b1 x ... bk) x d`-dim tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of the posterior variance of the model_varproxi
            at the given design points `X`.
        """
        model.to(device=X.device)  # ensures buffers / parameters are on the same device
        posterior = model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)  # removing redundant dimensions
        variance = posterior.variance.clamp_min(min_var).view(mean.shape)
        return mean, variance

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        self.beta = self.beta.to(X)
        batch_shape = X.shape[:-2]
        mean, variance = self._get_model_mean_and_sigma(X, self.model)
        delta = (self.beta.expand_as(variance) * variance).sqrt()

        self.beta_varproxi = self.beta_varproxi.to(X)
        mean_varproxi, variance_varproxi = self._get_model_mean_and_sigma(X, self.model_varproxi)
        delta_varproxi = (self.beta_varproxi.expand_as(variance_varproxi) * variance_varproxi).sqrt()

        # ucb = ucb_f - gamma*lcb_{rho}
        if self.maximize:
            return (mean + delta - self.gamma.expand_as(mean_varproxi) * (mean_varproxi - delta_varproxi))
        # lcb = lcb_f - gamma*ucb_{rho}
        else:
            return (mean - delta - self.gamma.expand_as(mean_varproxi) * (mean_varproxi + delta_varproxi))
