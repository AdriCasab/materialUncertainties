import gpytorch
import numpy as np
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
from testFunctions.MMW import MMWDeterministic
import os
from botorch.acquisition import LogExpectedImprovement

def boMain (runName, nCores, targetRadius, checkPoint):
    torch.set_default_dtype(torch.float64)
    GLOBAL_VARS = {
        'gamma': 1, # risk-tolerance constant for rahbo (corresponds to \alpha in the RAHBO paper)
        'sigma': 1, # noise variance in the area with maximum noise
        'seed': 0,  # random seed for initial evaluations generation
        'n_budget': 10, # number of BO iterations
        'repeat_eval': 5, # number of evaluations at the same point
        'beta': 2, # hyperparameter for UCB acquisition function
        'n_initial': 5, # number of initial points for BO (used to choose GP)
    }

    keepFiles = False
    simulationDirectory = "../steveEnthalpyModel/solver/Exec/run3d/"
    problem = MMWDeterministic(tarRadius=targetRadius, nCores=nCores, config=GLOBAL_VARS, simulationDirectory=simulationDirectory, compileStatus=False)

    
    outputFolder = 'results/' + str(runName)
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    outputFolder = 'results/' + str(runName) + '/data.txt'
    outputFolderLong = 'results/' + str(runName) + '/dataLong.txt'
    runTimeFolder = 'results/' + str(runName) + '/runTime.txt'
    if checkPoint == False:
        with open(outputFolder, 'w+') as f:
            f.write("\n")
        with open(outputFolderLong, 'w+') as f:
            f.write("\n")
        with open(runTimeFolder, 'w+') as f:
            f.write("\n")

    #define device
    tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    x = problem.get_random_initial_points(num_points=GLOBAL_VARS['n_initial'], seed=GLOBAL_VARS['seed'])
    y = problem.evaluate(x)
    print(x)
    print(y)

    ystd = y.std(dim=1, keepdim=True)
    ymean = y.mean(dim=1, keepdim=True)
    yMV = ymean - GLOBAL_VARS['gamma']*ystd**2

    for i in range(len(x)):
        with open(outputFolder, 'a') as f:
            f.write(str(x[i].tolist()) + " " + str(ymean[i].tolist()) + " " + str(ystd[i].tolist()) + " " + str(yMV[i].tolist()) +"\n")
        with open(outputFolderLong, 'a') as f:
            f.write(str(x[i].tolist()) + " " + str(y[i].tolist()) +"\n")

    bounds = problem.get_domain()
    for iteration in range(0, GLOBAL_VARS['n_budget']):
        #initialize GP model
        gp = SingleTaskGP(x, yMV,
                            input_transform=Normalize(d=problem.get_dim()), 
                            outcome_transform=Standardize(m=1), 
                            covar_module =MaternKernel(nu=2.5, 
                            lengthscale_prior=GammaPrior(3.0, 6.0))).to(x)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # gp_var = SingleTaskGP(x, ystd,  input_transform=Normalize(d=problem.get_dim()), 
        #                         outcome_transform=Standardize(m=1), 
        #                         covar_module =MaternKernel(nu=2.5, 
        #                         lengthscale_prior=GammaPrior(3.0, 6.0))).to(x)
        # mll_var = ExactMarginalLogLikelihood(gp_var.likelihood, gp_var)
        # fit_gpytorch_mll(mll_var)

        # acqf = RiskAverseUpperConfidenceBound(gp, gp_var, 
        #                                  beta=GLOBAL_VARS['beta'], beta_varproxi=GLOBAL_VARS['beta'],
        #                                  gamma=GLOBAL_VARS['gamma'])

        acqf = LogExpectedImprovement(gp, best_f=yMV.max())
        candidate, _ = optimize_acqf(
            acqf, bounds=bounds, q=1, num_restarts=10, raw_samples=512,
            )
        
        x = torch.cat([x, candidate])
        newObservation = problem.evaluate(candidate)
        y = torch.cat([y, newObservation])
        ymean = torch.cat([ymean, newObservation.mean(dim=1, keepdim=True)])
        ystd = torch.cat([ystd, newObservation.std(dim=1, keepdim=True)])
        yMV = ymean - GLOBAL_VARS['gamma']*ystd**2

        with open(outputFolder, 'a') as f:
            f.write(str(x[-1].tolist()) + " " + str(ymean[-1].tolist()) + " " + str(ystd[-1].tolist()) + " " + str(yMV[-1].tolist()) +"\n")
        with open(outputFolderLong, 'a') as f:
            f.write(str(x[-1].tolist()) + " " + str(y[-1].tolist()) +"\n")
        print("Completed iteration: ", iteration)

    

if __name__ == "__main__":
    runName = "test3"
    nCores = 16
    tarRadius = 0.01
    boMain(runName, nCores, tarRadius, False)