/*
This OpenCL kernel implements the classical Lax-Friedrichs scheme
for the shallow water equations, with edge fluxes.

Copyright (C) 2016  SINTEF ICT

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
*/


#include "common.cu"
#include "SWECommon.cu"
#include "fluxes/FirstOrderCentered.cu"


/**
  * Computes the flux along the x axis for all faces
  */
__device__ 
void computeFluxF(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
                  const float g_, const float dx_, const float dt_) {
                      
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Compute fluxes along the x axis
    {
        int j=ty;
        const int l = j + 1; //Skip ghost cells
        for (int i=tx; i<BLOCK_WIDTH+1; i+=BLOCK_WIDTH) {
            const int k = i;
            
            // Q at interface from the right and left
            const float3 Qp = make_float3(Q[0][l][k+1],
                                          Q[1][l][k+1],
                                          Q[2][l][k+1]);
            const float3 Qm = make_float3(Q[0][l][k],
                                          Q[1][l][k],
                                          Q[2][l][k]);
                                       
            // Computed flux
            const float3 flux = FORCE_1D_flux(Qm, Qp, g_, dx_, dt_);
            F[0][j][i] = flux.x;
            F[1][j][i] = flux.y;
            F[2][j][i] = flux.z;
        }
    }
}


/**
  * Computes the flux along the y axis for all faces
  */
__device__ 
void computeFluxG(float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2],
                  float G[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1],
                  const float g_, const float dy_, const float dt_) {
    //Index of thread within block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    //Compute fluxes along the y axis
    for (int j=ty; j<BLOCK_HEIGHT+1; j+=BLOCK_HEIGHT) {
        const int l = j;
        {
            int i=tx;
            const int k = i + 1; //Skip ghost cells
            
            // Q at interface from the right and left
            // Note that we swap hu and hv
            const float3 Qp = make_float3(Q[0][l+1][k],
                                          Q[2][l+1][k],
                                          Q[1][l+1][k]);
            const float3 Qm = make_float3(Q[0][l][k],
                                          Q[2][l][k],
                                          Q[1][l][k]);

            // Computed flux
            // Note that we swap back
            const float3 flux = FORCE_1D_flux(Qm, Qp, g_, dy_, dt_);
            G[0][j][i] = flux.x;
            G[1][j][i] = flux.z;
            G[2][j][i] = flux.y;
        }
    }
}


extern "C" {
__global__ void FORCEKernel(
        int nx_, int ny_,
        float dx_, float dy_, float dt_,
        float g_,
        
        //Input h^n
        float* h0_ptr_, int h0_pitch_,
        float* hu0_ptr_, int hu0_pitch_,
        float* hv0_ptr_, int hv0_pitch_,
        
        //Output h^{n+1}
        float* h1_ptr_, int h1_pitch_,
        float* hu1_ptr_, int hu1_pitch_,
        float* hv1_ptr_, int hv1_pitch_) {
    
    __shared__ float Q[3][BLOCK_HEIGHT+2][BLOCK_WIDTH+2];
    __shared__ float F[3][BLOCK_HEIGHT+1][BLOCK_WIDTH+1];
    
    
    //Read into shared memory
    readBlock1(h0_ptr_, h0_pitch_,
               hu0_ptr_, hu0_pitch_,
               hv0_ptr_, hv0_pitch_,
               Q, nx_, ny_);
    __syncthreads();
        
    
    //Set boundary conditions
    noFlowBoundary1(Q, nx_, ny_);
    __syncthreads();
    
    //Compute flux along x, and evolve
    computeFluxF(Q, F, g_, dx_, dt_);
    __syncthreads();
    evolveF1(Q, F, nx_, ny_, dx_, dt_);
    __syncthreads();
    
    //Set boundary conditions
    noFlowBoundary1(Q, nx_, ny_);
    __syncthreads();
    
    //Compute flux along y, and evolve
    computeFluxG(Q, F, g_, dy_, dt_);
    __syncthreads();
    evolveG1(Q, F, nx_, ny_, dy_, dt_);
    __syncthreads();
    
    //Write to main memory
    writeBlock1(h1_ptr_, h1_pitch_,
                hu1_ptr_, hu1_pitch_,
                hv1_ptr_, hv1_pitch_,
                Q, nx_, ny_);
}

} // extern "C"