# -*- coding: utf-8 -*-

"""
This python module implements MPI simulations for benchmarking

Copyright (C) 2018  SINTEF ICT

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
import os

# MPI
from mpi4py import MPI

# CUDA
import pycuda.driver as cuda

# Simulator engine etc
from GPUSimulators import MPISimulator, Common, CudaContext
from GPUSimulators import EE2D_KP07_dimsplit
from GPUSimulators.helpers import InitialConditions as IC
from GPUSimulators.Simulator import BoundaryCondition as BC

import argparse
parser = argparse.ArgumentParser(description='Strong and weak scaling experiments.')
parser.add_argument('-nx', type=int, default=128)
parser.add_argument('-ny', type=int, default=128)
parser.add_argument('--profile', action='store_true') # default: False


args = parser.parse_args()

if(args.profile):
    # profiling: total run time
    t_total_start = time.time()


# Get MPI COMM to use
comm = MPI.COMM_WORLD


####
# Initialize logging
####
log_level_console = 20
log_level_file = 10
log_filename = 'mpi_' + str(comm.rank) + '.log'
logger = logging.getLogger('GPUSimulators')
logger.setLevel(min(log_level_console, log_level_file))

ch = logging.StreamHandler()
ch.setLevel(log_level_console)
logger.addHandler(ch)
logger.info("Console logger using level %s",
            logging.getLevelName(log_level_console))

fh = logging.FileHandler(log_filename)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s: %(message)s')
fh.setFormatter(formatter)
fh.setLevel(log_level_file)
logger.addHandler(fh)
logger.info("File logger using level %s to %s",
            logging.getLevelName(log_level_file), log_filename)


####
# Initialize MPI grid etc
####
logger.info("Creating MPI grid")
grid = MPISimulator.MPIGrid(MPI.COMM_WORLD)


####
# Initialize CUDA
####
cuda.init(flags=0)
logger.info("Initializing CUDA")
local_rank = grid.getLocalRank()
num_cuda_devices = cuda.Device.count()
cuda_device = local_rank % num_cuda_devices
cuda_context = CudaContext.CudaContext(device=cuda_device, autotuning=False)


####
# Set initial conditions
####
logger.info("Generating initial conditions")
nx = args.nx
ny = args.ny

gamma = 1.4
save_times = np.linspace(0, 10.0, 2)
outfile = "mpi_out_" + str(MPI.COMM_WORLD.rank) + ".nc"
save_var_names = ['rho', 'rho_u', 'rho_v', 'E']

arguments = IC.genKelvinHelmholtz(nx, ny, gamma, grid=grid)
arguments['context'] = cuda_context
arguments['theta'] = 1.2
arguments['grid'] = grid


####
# Run simulation
####
logger.info("Running simulation")
# Helper function to create MPI simulator


def genSim(grid, **kwargs):
    local_sim = EE2D_KP07_dimsplit.EE2D_KP07_dimsplit(**kwargs)
    sim = MPISimulator.MPISimulator(local_sim, grid)
    return sim


outfile = Common.runSimulation(
    genSim, arguments, outfile, save_times, save_var_names)

if(args.profile):
    t_total_end = time.time()
    t_total = t_total_end - t_total_start
    print("Total run time on rank " + str(MPI.COMM_WORLD.rank) + " is " + str(t_total) + " s")

# write profiling to json file
if(args.profile and MPI.COMM_WORLD.rank == 0):
    if "SLURM_JOB_ID" in os.environ:
        job_id = int(os.environ["SLURM_JOB_ID"])
        allocated_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        allocated_gpus = int(os.environ["CUDA_VISIBLE_DEVICES"].count(",") + 1)
        profiling_file = "MPI_jobid_" + \
            str(job_id) + "_" + str(allocated_nodes) + "_nodes_and_" + str(allocated_gpus) + "_GPUs_profiling.json"
    else:
        profiling_file = "MPI_test_profiling.json"

    write_profiling_data = {}
    write_profiling_data["total"] = t_total

    with open(profiling_file, "w") as write_file:
        json.dump(write_profiling_data, write_file)

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
