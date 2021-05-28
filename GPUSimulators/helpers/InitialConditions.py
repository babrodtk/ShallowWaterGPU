# -*- coding: utf-8 -*-

"""
This python module implements Cuda context handling

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


from GPUSimulators.Simulator import BoundaryCondition
import numpy as np
import gc


def getExtent(width, height, nx, ny, grid, index=None):
    if grid is not None:
        gx = grid.grid[0]
        gy = grid.grid[1]
        if index is not None:
            i, j = grid.getCoordinate(index)
        else:
            i, j = grid.getCoordinate()
        
        dx = (width / gx) / nx
        dy = (height / gy) / ny
        
        x0 = width*i/gx + 0.5*dx
        y0 = height*j/gy + 0.5*dy
        x1 = width*(i+1)/gx - 0.5*dx
        y1 = height*(j+1)/gy - 0.5*dx
        
    else:
        dx = width / nx
        dy = height / ny
        
        x0 = 0.5*dx
        y0 = 0.5*dy
        x1 = width-0.5*dx
        y1 = height-0.5*dy
        
    return [x0, x1, y0, y1, dx, dy]

        
def downsample(highres_solution, x_factor, y_factor=None):
    if (y_factor == None):
        y_factor = x_factor

    assert(highres_solution.shape[1] % x_factor == 0)
    assert(highres_solution.shape[0] % y_factor == 0)
    
    if (x_factor*y_factor == 1):
        return highres_solution
    
    if (len(highres_solution.shape) == 1):
        highres_solution = highres_solution.reshape((1, highres_solution.size))

    nx = highres_solution.shape[1] / x_factor
    ny = highres_solution.shape[0] / y_factor

    return highres_solution.reshape([int(ny), int(y_factor), int(nx), int(x_factor)]).mean(3).mean(1)




    
def bump(nx, ny, width, height, 
        bump_size=None, 
        ref_nx=None, ref_ny=None,
        x_center=0.5, y_center=0.5,
        h_ref=0.5, h_amp=0.1, u_ref=0.0, u_amp=0.1, v_ref=0.0, v_amp=0.1):
    
    if (ref_nx == None):
        ref_nx = nx
    assert(ref_nx >= nx)
      
    if (ref_ny == None):
        ref_ny = ny
    assert(ref_ny >= ny)
        
    if (bump_size == None):
        bump_size = width/5.0
    
    ref_dx = width / float(ref_nx)
    ref_dy = height / float(ref_ny)

    x_center = ref_dx*ref_nx*x_center
    y_center = ref_dy*ref_ny*y_center
    
    x = ref_dx*(np.arange(0, ref_nx, dtype=np.float32)+0.5) - x_center
    y = ref_dy*(np.arange(0, ref_ny, dtype=np.float32)+0.5) - y_center
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    r = np.sqrt(xv**2 + yv**2)
    xv = None
    yv = None
    gc.collect()
    
    #Generate highres then downsample
    #h_highres = 0.5 + 0.1*np.exp(-(xv**2/size + yv**2/size))
    h_highres = h_ref + h_amp*0.5*(1.0 + np.cos(np.pi*r/bump_size)) * (r < bump_size)
    h = downsample(h_highres, ref_nx/nx, ref_ny/ny)
    h_highres = None
    gc.collect()
    
    #hu_highres = 0.1*np.exp(-(xv**2/size + yv**2/size))
    u_highres = u_ref + u_amp*0.5*(1.0 + np.cos(np.pi*r/bump_size)) * (r < bump_size)
    hu = downsample(u_highres, ref_nx/nx, ref_ny/ny)*h
    u_highres = None
    gc.collect()
    
    #hu_highres = 0.1*np.exp(-(xv**2/size + yv**2/size))
    v_highres = v_ref + v_amp*0.5*(1.0 + np.cos(np.pi*r/bump_size)) * (r < bump_size)
    hv = downsample(v_highres, ref_nx/nx, ref_ny/ny)*h
    v_highres = None
    gc.collect()
    
    dx = width/nx
    dy = height/ny
    
    return h, hu, hv, dx, dy


def genShockBubble(nx, ny, gamma, grid=None):
    """
    Generate Shock-bubble interaction case for the Euler equations
    """
    
    width = 4.0
    height = 1.0
    g = 0.0


    rho = np.ones((ny, nx), dtype=np.float32)
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    E = np.zeros((ny, nx), dtype=np.float32)
    p = np.ones((ny, nx), dtype=np.float32)

    
    x0, x1, y0, y1, dx, dy = getExtent(width, height, nx, ny, grid)
    x = np.linspace(x0, x1, nx, dtype=np.float32)
    y = np.linspace(y0, y1, ny, dtype=np.float32)
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
       
    #Bubble
    radius = 0.25
    x_center = 0.5
    y_center = 0.5
    bubble = np.sqrt((xv-x_center)**2+(yv-y_center)**2) <= radius
    rho = np.where(bubble, 0.1, rho)
    
    #Left boundary
    left = (xv < 0.1)
    rho = np.where(left, 3.81250, rho)
    u = np.where(left, 2.57669, u)
    
    #Energy
    p = np.where(left, 10.0, p)
    E = 0.5*rho*(u**2+v**2) + p/(gamma-1.0)
    

    bc = BoundaryCondition({
        'north': BoundaryCondition.Type.Reflective,
        'south': BoundaryCondition.Type.Reflective,
        'east': BoundaryCondition.Type.Periodic,
        'west': BoundaryCondition.Type.Periodic
    })
    
    #Construct simulator
    arguments = {
        'rho': rho, 'rho_u': rho*u, 'rho_v': rho*v, 'E': E,
        'nx': nx, 'ny': ny,
        'dx': dx, 'dy': dy, 
        'g': g,
        'gamma': gamma,
        'boundary_conditions': bc
    } 
    return arguments

    
    
    
    
    
    
def genKelvinHelmholtz(nx, ny, gamma, roughness=0.125, grid=None, index=None):
    """
    Roughness parameter in (0, 1.0] determines how "squiggly" 
    the interface betweeen the zones is
    """
    
    def genZones(nx, ny, n):
        """
        Generates the zones of the two fluids of K-H
        """
        zone = np.zeros((ny, nx), dtype=np.int32)


        def genSmoothRandom(nx, n):
            n = max(1, min(n, nx))
            
            if n == nx:
                return np.random.random(nx)-0.5
            else:
                from scipy.interpolate import interp1d

                #Control points and interpolator
                xp = np.linspace(0.0, 1.0, n)
                yp = np.random.random(n) - 0.5
                
                if (n == 1):
                    kind = 'nearest'
                elif (n == 2):
                    kind = 'linear'
                elif (n == 3):
                    kind = 'quadratic'
                else:
                    kind = 'cubic'
                    
                f = interp1d(xp, yp, kind=kind)

                #Interpolation points
                x = np.linspace(0.0, 1.0, nx)
                return f(x)



        x0, x1, y0, y1, _, dy = getExtent(1.0, 1.0, nx, ny, grid, index)
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
        _, y = np.meshgrid(x, y)

        #print(y+a[0])

        a = genSmoothRandom(nx, n)*dy
        zone = np.where(y > 0.25+a, zone, 1)

        a = genSmoothRandom(nx, n)*dy
        zone = np.where(y < 0.75+a, zone, 1)
        
        return zone
        
    width = 2.0
    height = 1.0
    g = 0.0
    gamma = 1.4

    rho = np.empty((ny, nx), dtype=np.float32)
    u = np.empty((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    p = 2.5*np.ones((ny, nx), dtype=np.float32)

    #Generate the different zones    
    zones = genZones(nx, ny, max(1, min(nx, int(nx*roughness))))
    
    #Zone 0
    zone0 = zones == 0
    rho = np.where(zone0, 1.0, rho)
    u = np.where(zone0, 0.5, u)
    
    #Zone 1
    zone1 = zones == 1
    rho = np.where(zone1, 2.0, rho)
    u = np.where(zone1, -0.5, u)
    
    E = 0.5*rho*(u**2+v**2) + p/(gamma-1.0)
    
    _, _, _, _, dx, dy = getExtent(width, height, nx, ny, grid, index)
    
    
    bc = BoundaryCondition({
        'north': BoundaryCondition.Type.Periodic,
        'south': BoundaryCondition.Type.Periodic,
        'east': BoundaryCondition.Type.Periodic,
        'west': BoundaryCondition.Type.Periodic
    })
    
    #Construct simulator
    arguments = {
        'rho': rho, 'rho_u': rho*u, 'rho_v': rho*v, 'E': E,
        'nx': nx, 'ny': ny,
        'dx': dx, 'dy': dy, 
        'g': g,
        'gamma': gamma,
        'boundary_conditions': bc
    } 
    
    return arguments
    
    
    
def genRayleighTaylor(nx, ny, gamma, version=0, grid=None):
    """
    Generates Rayleigh-Taylor instability case
    """
    width = 0.5
    height = 1.5
    g = 0.1

    rho = np.zeros((ny, nx), dtype=np.float32)
    u = np.zeros((ny, nx), dtype=np.float32)
    v = np.zeros((ny, nx), dtype=np.float32)
    p = np.zeros((ny, nx), dtype=np.float32)
    
    
    x0, x1, y0, y1, dx, dy = getExtent(width, height, nx, ny, grid)
    x = np.linspace(x0, x1, nx, dtype=np.float32)-width*0.5
    y = np.linspace(y0, y1, ny, dtype=np.float32)-height*0.5
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    
    #This gives a squigly interfact
    if (version == 0):
        y_threshold = 0.01*np.cos(2*np.pi*np.abs(x)/0.5)
        rho = np.where(yv <= y_threshold, 1.0, rho)
        rho = np.where(yv > y_threshold, 2.0, rho)
    elif (version == 1):
        rho = np.where(yv <= 0.0, 1.0, rho)
        rho = np.where(yv > 0.0, 2.0, rho)
        v = 0.01*(1.0 + np.cos(2*np.pi*xv/0.5))/4
    else:
        assert False, "Invalid version"
    
    p = 2.5 - rho*g*yv
    E = 0.5*rho*(u**2+v**2) + p/(gamma-1.0)
    
    bc = BoundaryCondition({
        'north': BoundaryCondition.Type.Reflective,
        'south': BoundaryCondition.Type.Reflective,
        'east': BoundaryCondition.Type.Reflective,
        'west': BoundaryCondition.Type.Reflective
    })
    
    #Construct simulator
    arguments = {
        'rho': rho, 'rho_u': rho*u, 'rho_v': rho*v, 'E': E,
        'nx': nx, 'ny': ny,
        'dx': dx, 'dy': dy, 
        'g': g,
        'gamma': gamma,
        'boundary_conditions': bc
    } 

    return arguments