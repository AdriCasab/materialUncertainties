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
from botorch.acquisition import UpperConfidenceBound, LogExpectedImprovement, LogNoisyExpectedImprovement
from acquisitionFunctions import RiskAverseUpperConfidenceBound
from botorch.optim import optimize_acqf

from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Normalize

from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.priors.torch_priors import GammaPrior
import time


torch.set_default_dtype(torch.float64)
GLOBAL_VARS = {
    'gamma': 1, # risk-tolerance constant for rahbo (corresponds to \alpha in the RAHBO paper)
    'sigma': 1, # noise variance in the area with maximum noise
    'seed': 0,  # random seed for initial evaluations generation
    'n_budget': 20, # number of BO iterations
    'n_budget_var': 10, # number of BO iterations to be used for variance learning (relevant only for RAHBO-US)
    'repeat_eval': 10, # number of evaluations at the same point
    'beta': 2, # hyperparameter for UCB acquisition function
    'n_initial': 10, # number of initial points for BO (used to choose GP)
    'n_bo_restarts': 10 # number of BO restarts (used for experiments plots in the paper)
}

#problem = SineBenchmark(GLOBAL_VARS)
problem = BraninTest(GLOBAL_VARS)
bounds=problem.get_domain()
#Considering naive UCB for only maximising expected mean

acquisitionFunctions = [UpperConfidenceBound, LogExpectedImprovement, LogNoisyExpectedImprovement, RiskAverseUpperConfidenceBound]
#acquisitionFunctions = [RiskAverseUpperConfidenceBound]
full_cumulative_regret = []
for acquisitionFunction in acquisitionFunctions:
    start = time.monotonic()
    acqf_cumulative_regret = None
    for restart in range(GLOBAL_VARS['n_bo_restarts']):
        x = problem.get_random_initial_points(num_points=GLOBAL_VARS['n_initial'], seed=GLOBAL_VARS['seed'] + restart)
        y = problem.evaluate(x)
        ystd = y.std(dim=1, keepdim=True)
        ymean = y.mean(dim=1, keepdim=True)
        yMV = ymean - GLOBAL_VARS['gamma']*ystd**2
        cumulative_regret = (problem.get_optimal_mv() - torch.max(problem.get_mv(x))).clone().detach().unsqueeze(0)
        yMV_max = yMV.max()
        #print(problem.get_optimal_mv())
        
        for iteration in range(0, GLOBAL_VARS['n_budget']):

            gp = SingleTaskGP(x, yMV, 
                                input_transform=Normalize(d=problem.get_dim()), 
                                outcome_transform=Standardize(m=1), 
                                covar_module =MaternKernel(nu=2.5, 
                                lengthscale_prior=GammaPrior(3.0, 6.0))).to(x)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            if acquisitionFunction == UpperConfidenceBound:
                acq = acquisitionFunction(gp, beta=GLOBAL_VARS['beta'])
            elif acquisitionFunction == LogNoisyExpectedImprovement:
                gp = SingleTaskGP(x, yMV, ystd,
                                input_transform=Normalize(d=problem.get_dim()), 
                                outcome_transform=Standardize(m=1), 
                                covar_module =MaternKernel(nu=2.5, 
                                lengthscale_prior=GammaPrior(3.0, 6.0))).to(x)
                acq = acquisitionFunction(gp, x)

            elif acquisitionFunction == RiskAverseUpperConfidenceBound:
                gp = SingleTaskGP(x, yMV,  
                                input_transform=Normalize(d=problem.get_dim()), 
                                outcome_transform=Standardize(m=1), 
                                covar_module =MaternKernel(nu=2.5, 
                                lengthscale_prior=GammaPrior(3.0, 6.0))).to(x)
                gp_var = SingleTaskGP(x, ystd,  input_transform=Normalize(d=problem.get_dim()), 
                                outcome_transform=Standardize(m=1), 
                                covar_module =MaternKernel(nu=2.5, 
                                lengthscale_prior=GammaPrior(3.0, 6.0))).to(x)
                mll_var = ExactMarginalLogLikelihood(gp_var.likelihood, gp_var)
                fit_gpytorch_mll(mll_var)
                acq = acquisitionFunction(gp, gp_var, 
                                         beta=GLOBAL_VARS['beta'], beta_varproxi=GLOBAL_VARS['beta'],
                                         gamma=GLOBAL_VARS['gamma'])
            else:
                acq = acquisitionFunction(gp, best_f=yMV_max)
            candidate, _ = optimize_acqf(
            acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512,
            )
            #print(candidate)
            x = torch.cat([x, candidate])
            newObservation = problem.evaluate(candidate)
            y = torch.cat([y, newObservation])
            ymean = torch.cat([ymean, newObservation.mean(dim=1, keepdim=True)])
            ystd = torch.cat([ystd, newObservation.std(dim=1, keepdim=True)])
            yMV = ymean - GLOBAL_VARS['gamma']*ystd**2
            yMV_max = yMV.max()
            cumulative_regret = torch.cat([cumulative_regret, cumulative_regret[-1] + (problem.get_optimal_mv() - torch.max(problem.get_mv(x))).unsqueeze(0)])
            
        if acqf_cumulative_regret == None:
            acqf_cumulative_regret = cumulative_regret
        else:
            if acqf_cumulative_regret.ndim == 1:
                acqf_cumulative_regret = torch.stack([acqf_cumulative_regret, cumulative_regret])
            else:
                cumulative_regret = cumulative_regret.unsqueeze(0)
                acqf_cumulative_regret = torch.cat([acqf_cumulative_regret, cumulative_regret], dim=0)                
    full_cumulative_regret.append(acqf_cumulative_regret)
    print(acquisitionFunction.__name__, time.monotonic()-start)
f, ax = plt.subplots(1, 1, figsize=(8, 6))

acqList = ["UCB", "EI", "NEI", "RAHBO"]
#acqList = ["RAHBO"]
for i in range(len(acqList)):
    median = np.median(full_cumulative_regret[i], axis=0)
    q1 = np.quantile(full_cumulative_regret[i], 0.25, axis=0)
    q3 = np.quantile(full_cumulative_regret[i], 0.75, axis=0)
    ax.plot(np.arange(1, len(median)+1), median, label = acqList[i])
    ax.fill_between(range(1,len(median)+1), q1, q3, alpha = 0.3)

ax.set_title('Acquisition Function Comparision')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cumulative Regret')
ax.legend(loc='best', framealpha=0.2)
plt.savefig('CumulativeRegret.png', dpi = 300)
