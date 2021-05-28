# -*- coding: utf-8 -*-

"""
This python module implements SHMEM (shared memory) simulations for benchmarking

Copyright (C) 2020 Norwegian Meteorological Institute

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
import gc
import time
import json
import logging

#Simulator engine etc
from GPUSimulators import SHMEMSimulatorGroup, Common, CudaContext
from GPUSimulators import EE2D_KP07_dimsplit
from GPUSimulators.helpers import InitialConditions as IC
from GPUSimulators.Simulator import BoundaryCondition as BC



####
#Initialize logging 
####
log_level_console = 20
log_level_file    = 10
log_filename = 'shmem.log'
logger = logging.getLogger('GPUSimulators')
logger.setLevel(min(log_level_console, log_level_file))

ch = logging.StreamHandler()
ch.setLevel(log_level_console)
logger.addHandler(ch)
logger.info("Console logger using level %s", logging.getLevelName(log_level_console))

fh = logging.FileHandler(log_filename)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(levelname)s: %(message)s')
fh.setFormatter(formatter)
fh.setLevel(log_level_file)
logger.addHandler(fh)
logger.info("File logger using level %s to %s", logging.getLevelName(log_level_file), log_filename)



####
# Initialize SHMEM grid etc
####

logger.info("Creating SHMEM grid")

# XXX: need to explicitly set ngpus when testing on single-GPU system
grid = SHMEMSimulatorGroup.SHMEMGrid(ngpus=4) 



####
# Set initial conditions
####
logger.info("Generating initial conditions")
nx = 128
ny = 128
gamma = 1.4
#save_times = np.linspace(0, 0.01, 10)
save_times = np.linspace(0, 10, 10)
save_var_names = ['rho', 'rho_u', 'rho_v', 'E']

outfile = "shmem_out.nc"

####
# Run simulation
####
logger.info("Running simulation")

sims = []
for i in range(grid.ngpus):
    arguments = IC.genKelvinHelmholtz(nx, ny, gamma, grid=grid, index=i)
    arguments['context'] = grid.cuda_contexts[i]
    arguments['theta'] = 1.2

    sims.append(EE2D_KP07_dimsplit.EE2D_KP07_dimsplit(**arguments))
    #sims[i] = SHMEMSimulator(i, local_sim, grid) # 1st attempt: no wrapper (per sim)

arguments['sims'] = sims
arguments['grid'] = grid

#Helper function to create SHMEM simulator
def genSim(sims, grid, **kwargs):
    # XXX: kwargs not used, since the simulators are already instantiated in the for-loop above
    sim = SHMEMSimulatorGroup.SHMEMSimulatorGroup(sims, grid)
    return sim

outfile = Common.runSimulation(genSim, arguments, outfile, save_times, save_var_names)



####
# Clean shutdown
####
sim = None
local_sim = None
cuda_context = None
arguments = None
logging.shutdown()
gc.collect()



####
# Print completion and exit
####
print("Completed!")
exit(0)