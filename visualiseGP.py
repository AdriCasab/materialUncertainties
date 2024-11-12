import gpytorch
import numpy as np
from testFunctions.branin import BraninTest
from testFunctions.sine import SineBenchmark
import torch
from botorch.models import SingleTaskGP
from botorch.models import HeteroskedasticSingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import matplotlib.pyplot as plt
from botorch.acquisition import LogNoisyExpectedImprovement
from botorch.optim import optimize_acqf

from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior

torch.set_default_dtype(torch.float64)
GLOBAL_VARS = {
    'gamma': 1, # risk-tolerance constant for rahbo (corresponds to \alpha in the RAHBO paper)
    'sigma': 1, # noise variance in the area with maximum noise
    'seed': 0,  # random seed for initial evaluations generation
    'n_budget': 5, # number of BO iterations
    'n_budget_var': 100, # number of BO iterations to be used for variance learning (relevant only for RAHBO-US)
    'repeat_eval': 100, # number of evaluations at the same point
    'beta': 2, # hyperparameter for UCB acquisition function
    'n_initial': 10, # number of initial points for BO (used to choose GP)
    'n_bo_restarts': 1 # number of BO restarts (used for experiments plots in the paper)
}


problem = SineBenchmark(GLOBAL_VARS)
x = problem.get_random_initial_points(num_points=GLOBAL_VARS['n_initial'], seed=GLOBAL_VARS['seed'])
y = problem.evaluate(x)

ystd = y.std(dim=1, keepdim=True)
ymean = y.mean(dim=1, keepdim=True)
ystd_true = problem.get_noise_var(x)

#Fitting the mean model
gp = SingleTaskGP(x, ymean, 
                    input_transform=Normalize(d=1), 
                    outcome_transform=Standardize(m=1), 
                    covar_module =MaternKernel(nu=2.5, 
                    lengthscale_prior=GammaPrior(3.0, 6.0)) ).to(x)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x_1 = torch.linspace(0, 1, 100).view(-1, 1)
    test_x = torch.cat([test_x_1, test_x_1+1])
    observed_pred = gp.posterior(test_x)
    true_vals = problem.evaluate_true(test_x)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ymin = -3
    ymax = 3

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x.numpy(), ymean.numpy(), 'o', color ='k', label='Observed Data')
    ax.vlines(x=1, ymin=ymin, ymax=ymax, colors ='g', label='Variance Shift')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Mean Prediction')
    ax.plot(test_x.numpy(), true_vals.numpy(), 'r', label='True Solution')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy().reshape((len(test_x),)), lower.numpy(), upper.numpy(), alpha=0.5, label='Two Sigma Confidence Region')
    ax.set_ylim([ymin, ymax])
    ax.set_title('Fitted Mean model')
    ax.legend(loc='best', framealpha=0.2)
    plt.savefig('gpfit/fitted_model.png', dpi =200)

#Fitting the mean model

gp_std = SingleTaskGP(x, ystd,
                    input_transform=Normalize(d=1), 
                    outcome_transform=Standardize(m=1), 
                    covar_module =MaternKernel(nu=2.5, 
                    lengthscale_prior=GammaPrior(3.0, 6.0)) ).to(x)
mll_std = ExactMarginalLogLikelihood(gp_std.likelihood, gp_std)
fit_gpytorch_mll(mll_std)


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x_1 = torch.linspace(0, 1, 100).view(-1, 1)
    test_x = torch.cat([test_x_1, test_x_1+1])
    observed_pred = gp_std.posterior(test_x)
    true_vals = problem.get_noise_var(test_x) #convert from std to variance

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ymin = -1
    ymax = 2

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x.numpy(), ystd.numpy(), 'o', color ='k', label='Observed Data')
    ax.vlines(x=1, ymin=ymin, ymax=ymax, colors ='g', label='Variance Shift')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Mean Prediction')
    ax.plot(test_x.numpy(), true_vals.numpy(), 'r', label='True Solution')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy().reshape((len(test_x),)), lower.numpy(), upper.numpy(), alpha=0.5, label='Two Sigma Confidence Region')
    ax.set_ylim([ymin, ymax])
    ax.set_title('Fitted Variance model')
    ax.legend(loc='best', framealpha=0.2)
    plt.savefig('gpfit/fitted_stdModel.png', dpi = 200)

yMV = ymean - GLOBAL_VARS['gamma']*ystd**2
gp_MV = SingleTaskGP(x, yMV, 
                    input_transform=Normalize(d=1), 
                    outcome_transform=Standardize(m=1), 
                    covar_module =MaternKernel(nu=2.5, 
                    lengthscale_prior=GammaPrior(3.0, 6.0)) ).to(x)
mll_MV = ExactMarginalLogLikelihood(gp_MV.likelihood, gp_MV)
fit_gpytorch_mll(mll_MV)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x_1 = torch.linspace(0, 1, 100).view(-1, 1)
    test_x = torch.cat([test_x_1, test_x_1+1])
    observed_pred = gp_MV.posterior(test_x)
    true_vals = problem.get_mv(test_x) #convert from std to variance

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ymin = -3
    ymax = 3

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(x.numpy(), yMV.numpy(), 'o', color ='k', label='Observed Data')
    ax.vlines(x=1, ymin=ymin, ymax=ymax, colors ='g', label='Variance Shift')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Mean Prediction')
    ax.plot(test_x.numpy(), true_vals.numpy(), 'r', label='True Solution')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy().reshape((len(test_x),)), lower.numpy(), upper.numpy(), alpha=0.5, label='Two Sigma Confidence Region')
    ax.set_ylim([ymin, ymax])
    ax.set_title('Fitted Mean-Variance model - One GP')
    ax.legend(loc='best', framealpha=0.2)
    plt.savefig('gpfit/fitted_MVModel_1GP.png', dpi = 200)


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x_1 = torch.linspace(0, 1, 100).view(-1, 1)
    test_x = torch.cat([test_x_1, test_x_1+1])
    observed_mean = gp.posterior(test_x)
    observed_std = gp_std.posterior(test_x)
    observed_MV = observed_mean.mean.numpy() - GLOBAL_VARS['gamma']*observed_std.mean.numpy()**2 #obseved mean-variance model
    variance_MV = observed_mean.variance.numpy() + GLOBAL_VARS['gamma']**2*observed_std.variance.numpy()**2 #obseved mean-variance model
    true_vals = problem.get_mv(test_x) #convert from std to variance
    print(problem.get_optimal_mv())

yMV = ymean - GLOBAL_VARS['gamma']*ystd**2
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ymin = -3
    ymax = 3

    # Get upper and lower confidence bounds
    lower, upper = observed_MV-2*np.sqrt(variance_MV), observed_MV+2*np.sqrt(variance_MV)
    # Plot training data as black stars
    ax.plot(x.numpy(), yMV.numpy(), 'o', color ='k', label='Observed Data')
    ax.vlines(x=1, ymin=ymin, ymax=ymax, colors ='g', label='Variance Shift')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_MV, 'b', label='Mean Prediction')
    ax.plot(test_x.numpy(), true_vals.numpy(), 'r', label='True Solution')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy().reshape((len(test_x),)), lower.reshape(-1), upper.reshape(-1), alpha=0.5, label='Two Sigma Confidence Region')
    ax.set_ylim([ymin, ymax])
    ax.set_title('Fitted Mean-Variance model - Mean GP and Variance GP')
    ax.legend(loc='best', framealpha=0.2)
    plt.savefig('gpfit/fitted_MVModel_2GPs.png', dpi = 200)

naive_UCB_cumulative_regret_list = None

bounds=problem.get_domain()
for iteration in range(0, GLOBAL_VARS['n_budget']):
    # Fit the models
    gp = SingleTaskGP(x, yMV, ystd**2, input_transform=Normalize(d=1), outcome_transform=Standardize(m=1),
                      covar_module =MaternKernel(nu=2.5, 
                    lengthscale_prior=GammaPrior(3.0, 6.0)) ).to(x)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    ucb = LogNoisyExpectedImprovement(gp, x)

    candidate, _ = optimize_acqf(
    ucb, bounds=bounds, q=1, num_restarts=10, raw_samples=512,
    )
    x = torch.cat([x, candidate])
    newObservation = problem.evaluate(candidate)
    y = torch.cat([y, newObservation])
    ymean = torch.cat([ymean, newObservation.mean(dim=1, keepdim=True)])
    ystd = torch.cat([ystd, newObservation.std(dim=1, keepdim=True)])
    yMV = ymean - GLOBAL_VARS['gamma']*ystd**2

gp = SingleTaskGP(x, ymean, 
                    input_transform=Normalize(d=1), 
                    outcome_transform=Standardize(m=1), 
                    covar_module =MaternKernel(nu=2.5, 
                    lengthscale_prior=GammaPrior(3.0, 6.0)) ).to(x)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)


gp_std = SingleTaskGP(x, ystd, 
                    input_transform=Normalize(d=1), 
                    outcome_transform=Standardize(m=1), 
                    covar_module =MaternKernel(nu=2.5, 
                    lengthscale_prior=GammaPrior(3.0, 6.0)) ).to(x)
mll_std = ExactMarginalLogLikelihood(gp_std.likelihood, gp_std)
fit_gpytorch_mll(mll_std)


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x_1 = torch.linspace(0, 1, 100).view(-1, 1)
    test_x = torch.cat([test_x_1, test_x_1+1])
    observed_mean = gp.posterior(test_x)
    observed_std = gp_std.posterior(test_x)
    observed_MV = observed_mean.mean.numpy() - GLOBAL_VARS['gamma']*observed_std.mean.numpy()**2 #obseved mean-variance model
    variance_MV = observed_mean.variance.numpy() + GLOBAL_VARS['gamma']**2*observed_std.variance.numpy()**2 #obseved mean-variance model
    true_vals = problem.get_mv(test_x) #convert from std to variance
    print(problem.get_optimal_mv())

yMV = ymean - GLOBAL_VARS['gamma']*ystd**2
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 6))
    ymin = -3
    ymax = 3

    # Get upper and lower confidence bounds
    lower, upper = observed_MV-2*np.sqrt(variance_MV), observed_MV+2*np.sqrt(variance_MV)
    # Plot training data as black stars
    ax.plot(x.numpy(), yMV.numpy(), 'o', color ='k', label='Observed Data')
    ax.vlines(x=1, ymin=ymin, ymax=ymax, colors ='g', label='Variance Shift')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_MV, 'b', label='Mean Prediction')
    ax.plot(test_x.numpy(), true_vals.numpy(), 'r', label='True Solution')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy().reshape((len(test_x),)), lower.reshape(-1), upper.reshape(-1), alpha=0.5, label='Two Sigma Confidence Region')
    ax.set_ylim([ymin, ymax])
    ax.set_title('Fitted Mean-Variance model')
    ax.legend(loc='best', framealpha=0.2)
    plt.savefig('gpfit/fitted_MVModelPostBO.png', dpi = 200)