from botorch.test_functions.base import BaseTestProblem
from typing import List, Tuple, Union
import os
import torch
from torch import Tensor
import time
import yt
import numpy as np
from botorch.utils.sampling import draw_sobol_samples

class MMWDeterministic(BaseTestProblem):
    """
    Basic Problem for MMW Ablation Test. This excludes the stochastic noise induced from material properties in MMW ablation.
    The noise is manually added as a function of the input power. 
    In addition there is are no constraints.

    The purpose of this function is for the development of the BO loop for MMW ablation.

    List of functions in this class:
    1. __init__: Constructor for the class
    2. evaluate_true: Function to evaluate the true value of the function without noise
    3. evaluate: Function to embed noise in the result of the simulation

    """
    dim = 2 #[Power, Beam Radius]
    num_objectives = 1 #[Rate of Penetration]
    _bounds = [(20000, 50000), (0.001, 0.03)] #Power: 20kW to 50kW, Beam Radius: 1mm to 5cm
    _simDirectory = None #directory where the simulation is located and ran
    _settingsFile = None #name of settings file for AMReX

    standOffDistance = 0.1524 #considering fixed stand-off distance

    def __init__(self, tarRadius: float, nCores: int, config: dict, simulationDirectory: str = None, noise_std: Union[None, float, List[float]] = None, negate: bool = False, compileStatus: bool = True, settingsFile: str = None) -> None:
        super().__init__(noise_std=noise_std, negate=negate)
        self.targetRadius = tarRadius
        self.nCores = nCores 
        self.sigma = config['sigma']
        self.repeat_eval = config['repeat_eval']
        self.gamma = config['gamma']
        

        #setting simlation directory
        if simulationDirectory is None:
            simulationDirectory = "../solver/Exec/run3d/"
            print("setting simulation directory to standard directory")
        self._simDirectory = simulationDirectory
        #attempting to compile AMReX based solver
        if compileStatus:
            makeCall = "cd  " + str(simulationDirectory) +" && make -j"  + str(nCores)
            os.system(makeCall)
        if settingsFile == None:
            self._settingsFile = "settingsAdri.cmp"
    

    def evaluate_true(self, X: Tensor) -> Tensor:
        drillDepth = []
        if X.dim() == 1:
            X = X.unsqueeze(0)
        for x in X:
            runStart = time.time()
            power, radius = x.unbind(dim=-1)
            print(power.cpu().detach().numpy(), radius.cpu().detach().numpy())
            depth = self.wrapperFunction(power.cpu().detach().numpy(), radius.cpu().detach().numpy(), self.standOffDistance, self.targetRadius, nCores = self.nCores, keepFiles = True)
            drillDepth.append(depth)
            print("drill depth: " + str(depth))
            print("simulation compute time: " + str(time.time() - runStart))
        print(drillDepth)
        return torch.tensor(drillDepth).squeeze()

    def evaluate(self, x: Tensor, seed_eval=None) -> Tensor:
        y_true = self.evaluate_true(x).reshape((-1, 1))
        print(y_true)
        sigmas = self.get_noise_var(x).reshape((-1, 1))

        if seed_eval is not None:
            shape = torch.cat([y_true] * self.repeat_eval, dim=1).shape
            noise = sigmas * torch.randn(shape, generator=torch.Generator().manual_seed(seed_eval))
            y = y_true + noise
        else:
            noise = sigmas * torch.randn_like(torch.cat([y_true] * self.repeat_eval, dim=1))
            y = y_true + noise
        return y
    
    def get_noise_var(self, X: Tensor) -> Tensor:
        sigma = []
        if X.dim() == 1:
            X = X.unsqueeze(0)
        for x in X:
            power, radius = x.unbind(dim=-1)
            sigma.append((power/1e5)**2*5e-1)
        return torch.tensor(sigma).squeeze()
    
    def callSim(self, power: float, radius: float,  standOffDistance: float, targetRadius: float, nCores:int) -> None:
        """
        This function calls the AMReX function with the specified inputs
        """
        nCells = 64
        nCellsStr = " amr.n_cell = " + str(nCells) + " " + str(nCells) + " "  + str(nCells)

        #to be revisted for not TEM00
        maxZ = 0.01 * 60 *power * 0.7 * 0.5 /(26000 *3.14159 * (radius+0.01)**2*10000) #rule from Oglesby eq 2 and 3
        maxZ *= 5 #arbitrary factor to ensure that the domain is large enough
        maxZ = max(maxZ, targetRadius*2.5)
        xyDimension = maxZ
        timeStepStr = " he.coarseDt = " + str( 5*100000.0*(float(radius))/(float(power)))

        #converting inputs to string to run in the osCall
        powerStr = " he.laser_power = "+ str(power)
        radiusStr = " he.beam_radius = " + str(radius)
        geomLoStr = " geometry.prob_lo = "  + str(0.0) + " " + str(0.0) + " " + str(-maxZ)
        posDz = maxZ/(nCells*1.5)
        geomHiStr = " geometry.prob_hi = "  + str(xyDimension) + " " + str(xyDimension) + " " + str(posDz)
        standOffStr = " he.stand_off_distance = " + str(standOffDistance)
        
        #calling the AMReX solver
        osCall = "mpirun -n " + str(nCores) + " ./main3d.gnu.MPI.ex " + self._settingsFile + nCellsStr + powerStr + radiusStr +  standOffStr  + geomLoStr + geomHiStr  + timeStepStr +" > /dev/null"
        print(osCall)
        os.system("cd " + self._simDirectory + " && " + osCall)

    def clearResults(self) -> None:
        """
        This function clears the results directory to limit memory usage
        """
        os.system("rm -r "  +str(self._simDirectory) + "/results")

    def wrapperFunction(self, power: float, radius: float, standOffDistance: float, targetRadius: float, nCores: int, keepFiles: bool) -> float:
        """
        This function wraps the AMReX solver in python, to render the function usable for BoTorch.

        Args:
            targetRadius: intended radius of the wellbore
            keepFiles: boolean to specify whether to keep the results files or not

        Returns:
            Rate of penetration
            Thermal Specific Efficiency: power/total volume removed
            Wellbore Deviation
        """
        self.clearResults()
        self.callSim(power, radius, standOffDistance, targetRadius, nCores)
        drillDepth = self.readDirectoryLast(targetRadius, str(self._simDirectory) + "/results/")
        
        if keepFiles == False:
            self.clearResults()
        return drillDepth
    
    def readDirectoryLast(self, minRadius: str, targetDir: str) -> Tuple[float, float, float]:
        """
        This function calls readResults and reads the results of the last timestep within the directory

        Args:
            targetDir: target directory containing all files of the simulation
        """    
        directoryFiles = os.listdir(targetDir)
        directoryFiles.sort()
        drillDepth = self.readResults(minRadius=minRadius, resultsDir = targetDir + directoryFiles[-1])

        return drillDepth
    
    def readResults(self, minRadius: float, resultsDir: str) -> Tuple[float, float, float]:
        """
        This function reads the results files in the target directory and outputs the relevant measurement values. 
        For the MMW drilling study these are drilled depth, total volume removed (later used to compute the thermal specific efficiency), and wellboreDeviation

        Args:
            minRadius: The target diameter of the wellbore
            resultsDir: the directory of the files to be read (e.g. "Exec/run3d/results/plt0030")
        Returns:
            drilled depth
            total volume removed (multiplied by 4 as only a quarter of the domain is simulated)
            thermal specific energy
        """

        ds = yt.load(resultsDir)
        data = ds.all_data()
        print(dir(ds.fields.boxlib))
        print(dir(ds.fields.gas))
        print(dir(ds.fields.index))

        #initialize outputs
        drillDepth = 0
        
        #preemptively convert some of the ytarrays to numpy for faster processing
        zCoordinate= data["boxlib", "z"].value
        vapFraction = data["boxlib", "VapourVolumeFraction"].value
        dxCell  = data["boxlib", "dx"].value
        dyCell  = data["boxlib", "dy"].value
        dzCell  = data["boxlib", "dz"].value
        cellVolume = data["boxlib", "cell_volume"].value
        numCells = zCoordinate.size
        print("number of cells : " + str(numCells))

        xCoordinate = data["boxlib", "x"].value
        yCoordinate = data["boxlib", "y"].value
        radialCoordinate = np.sqrt(xCoordinate**2 + yCoordinate**2)

        #computing maximum depthed drilled where the wellbore radius is above the specified dimension by minRadius
        drillDepth = 0
        validDepths = []

        for i in range(numCells):
            if vapFraction[i] > 0.9:
                validDepths.append(zCoordinate[i])

        invalidDepths = []        
        
        validDepths = set(validDepths)
        for i in range(numCells):
            if vapFraction[i] < 0.9 and radialCoordinate[i] < minRadius:
                validDepths.discard(zCoordinate[i])
                invalidDepths.append(zCoordinate[i])
        invalidDepths = set(invalidDepths)
        validDepths = {x for x in validDepths if all(x > y for y in invalidDepths)}
        drillDepth = abs(min(validDepths))
        
        if drillDepth < min(dzCell):
            drillDepth = 0.0
        
        return drillDepth

    def get_domain(self):
        return torch.Tensor(self._bounds).T
    
    def get_dim(self):
        return self.dim
    
    def get_random_initial_points(self, num_points, seed) -> Tensor:

        x = draw_sobol_samples(self.get_domain(), num_points, q=1, seed=seed).squeeze()
        return x
    
    
    
