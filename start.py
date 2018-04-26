#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Start of final project: how do faults, fans, and channels evolve together?

SOURCE ACTIVATE LANDLAB FIRST!!!

Created on Wed Apr 18 12:47:22 2018

@author: nadine
"""
#%% import modules
import numpy as np
np.random.seed(2090)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from landlab import RasterModelGrid, imshow_grid 
from landlab.components import LinearDiffuser, DepthDependentDiffuser, ExponentialWeatherer
from landlab.components import FlowAccumulator, FlowRouter, FastscapeEroder
from landlab.plot.drainage_plot import drainage_plot

def surf_plot(mg, surface='topographic__elevation', zlimits=None, title='Surface plot of topography'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = mg.at_node[surface].reshape(mg.shape)
    color = cm.terrain((Z-Z.min())/(Z.max()-Z.min())) # replace gray with terrain
    ax.plot_surface(mg.node_x.reshape(mg.shape), mg.node_y.reshape(mg.shape),Z, 
                    rstride=1, cstride=1, facecolors=color, linewidth=0., antialiased=False)
    ax.view_init(elev=45, azim=-115)
    ax.set_zlim(zlimits)
    ax.set_xlabel('X axis (m)')
    ax.set_ylabel('Y axis (m)')
    ax.set_zlabel('Elevation (m)')
    plt.title(title)
    plt.show()

#%% initialize / set parameters and boundary conditions

###### initialize grid ######
x = 100
y = 50
spacing = 1.0 # meters
grid = RasterModelGrid((y,x+1),spacing) # rows,columns,spacing

###### set boundary conditions on grid ######
# east, north, west, south,  True = closed; False = open
grid.set_closed_boundaries_at_grid_edges(True, True, True, False)

###### time ######
dt = 10.0 # years
tmax = 1000 # years
time = np.arange(0,tmax,dt)

###### topography ######
zb = grid.add_zeros('node', 'bedrock__elevation') # add bedrock topography (zero everywhere) to grid
zb[:] += grid.node_y * 0.1 # add ramp in y direction
h = grid.add_ones('node','soil__depth') # add soil (1 m everywhere)
h[:] += 100 # soil/alluvium thickness in meters
z = grid.add_zeros('node','topographic__elevation') # add empty topography elevation
initial_roughness = np.random.rand(zb.size)/10. # make random noise to generate channels
zb[:] += initial_roughness # add random noise to bedrock
z[:] = h[:] + zb[:] # set topo elevation to be sum of soil and bedrock elevations
channel_loc = (x/2) - 1
z[np.where(grid.node_x == channel_loc)] -= 10
z[np.where(grid.node_x == channel_loc+1)] -= 10

###### geomorph params ######
hstar = 0.5 # scaling parameter for weathering rate...characteristic depth
Wo = 1e-6 # initial weathering rate mm/yr [meters/yr]
#Wo = 1e-3
rhos = 1.33 * 1e3 #1.33 * 1e6 # density soil 1.33 g/cm3 --> [g/m^3]
rhor = 2.65 * 1e3 # 2.65 * 1e6 # density rock 2.65 g/cm3  --> [g/m^3]
kappa = 0.01  # m2/yr diffusivity / transport coefficient
kappa = 0.005
ideal_dt = 0.5 * grid.dx * grid.dx / kappa # calculate ideal dt based on relationship we learned about (CFL)


###### tectonics ######
fault_loc = (y / 2) - 1
fault_nodes = np.where(grid.node_y == fault_loc)[0]

slip_interseis = 0 # magnitude not taken into account!!!
slip_seis = 1 # magnitude not taken into account!!!
slip_freq = 250 / dt
slip_freq_high = 100/dt
slip = np.zeros(len(time))
slip[:] += slip_interseis
slip[10:-1:slip_freq_high] = slip_seis


###### plot initial conditions ######
plt.figure(figsize=(8,4))
imshow_grid(grid,z,limits=(90,110),cmap='terrain',grid_units=['m','m'],shrink=.6)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--')
plt.title('Initial Topography')
plt.show()

plt.figure(figsize=(6,4))
plt.plot(time, slip)
plt.title('Tectonic Regime')
plt.xlabel('time (years)')
plt.ylabel('slip one node (on or off)')
plt.show()

print('ideal dt is <'+str(ideal_dt)+' years')
print('dt is '+ str(dt)+' years')
print('running for '+str(tmax)+' years')

######## initialize landlab components ############
linear_diffuser = LinearDiffuser(grid,linear_diffusivity=kappa)
depth_diffuser = DepthDependentDiffuser(grid,linear_diffusivity=kappa,soil_transport_decay_depth=0.3)
weatherer = ExponentialWeatherer(grid,max_soil_production_rate=Wo, soil_production_decay_depth=hstar)
flow_router = FlowRouter(grid) # optional parameters: method='D8',runoff_rate=None)
flow_accumulator = FlowAccumulator(grid,surface='topographic__elevation',flow_director='FlowDirectorD8',
                     depression_finder='DepressionFinderAndRouter') # FlowDirector D8 or Steepest
fastscape_eroder = FastscapeEroder(grid,K_sp=.002, m_sp=0.5, n_sp=1., threshold_sp=0.) #rainfall_intensity=1.)
## GET SPACE IN HERE!


#%%  RUN 

#iterations = 20
#iterations = np.int(time/dt) + 1
plots = 2

fig = plt.figure(figsize=(8,4))
writer = animation.FFMpegWriter(fps=5) #optional: extra_args=['-vcodec', 'libx264']
writer.setup(fig, 'test.mp4', dpi=300)
imshow_grid(grid,z,limits=(90,110),cmap='terrain',grid_units=['m','m'],shrink=.7)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--')
plt.title('Topography after 0 years')
writer.grab_frame()

for i in range(len(time)):
    
    #linear_diffuser.run_one_step(dt)
    #weatherer.calc_soil_prod_rate() #WHAT DOES THIS DO?
    weatherer.run_one_step(dt)
    depth_diffuser.run_one_step(dt)
    #flow_router.run_one_step()
    flow_accumulator.run_one_step()
    fastscape_eroder.run_one_step(dt)
    zb[:] = np.minimum(zb[:],z[:])     #find where zb>z, make it z so that bedrock never goes above topo
    h[:] = z[:] - zb[:]

    if slip[i] > 0:
#        z[fault_nodes] = z[grid.neighbors_at_node[fault_nodes][0]] # DOES NOT WORK 
        # TRY RESHAPE??
        for node in range(len(z)):
            if grid.node_y[node] < fault_loc:
                z[node] = z[grid.neighbors_at_node[node][0]]
        ## I DON"T APPLY ANYTHING TO NEW Z CELLS, BUT SOMEHOW THEY HAVE VALUES

    if i % plots == 0:
        plt.clf()
        imshow_grid(grid,z,limits=(90,110),cmap='terrain',grid_units=['m','m'],shrink=.7)
        plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--')
        plt.title('Topography after '+str(np.int((i*dt)+dt))+' years')
        writer.grab_frame()

plt.clf()        
imshow_grid(grid,z,limits=(90,110),cmap='terrain',grid_units=['m','m'],shrink=.7)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--')
plt.title('Topography after '+str(np.int((i*dt)+dt))+' years')
writer.grab_frame()
plt.show()

writer.finish()


#%%  other plots

### show grid for testing locations
#plt.plot(grid.node_x,grid.node_y,'.') 
#plt.plot(grid.nodes[0,10],'r.')

### 3D plot of topography
surf_plot(grid,surface='topographic__elevation',title='Topographic Elevation')
