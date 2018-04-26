#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Strike-Slip tectonic model

as part of final project: how do faults, fans, and channels evolve together?

SOURCE ACTIVATE LANDLAB FIRST!!!

# updated to only fault core nodes
# updated to add realistic z to new nodes on right side in slip events

Created on Wed Apr 18 12:47:22 2018

@author: nadine
"""
#%% THINGS TO DO:

# magntiude of offset change implementation
# kappa and rho values for DV alluvium ?
# implement storm events / climate regimes
# implement shear zone (OFD) instead of 1 fault


#%% THINGS I WANT IN A FUNCTION
# - erosion on or off
# - erosion scheme (FR, FA, ErosionDeposition)
# - kappa values
# - starting topo: flat, ramp, fan, etc
# - slip magntiude
# - slip interval
# - grid size (X,Y)
# - initial topo complexity (number and spacing of channels/channel network)
# - make topo or load topo values


#%% import modules
import numpy as np
np.random.seed(2092)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from landlab import RasterModelGrid, imshow_grid, FIXED_VALUE_BOUNDARY
from landlab.components import LinearDiffuser, DepthDependentDiffuser, ExponentialWeatherer, DepressionFinderAndRouter
from landlab.components import FlowAccumulator, FlowRouter, FastscapeEroder, ErosionDeposition
#from landlab.components import ErosionDeposition
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
x = 500
y = 250
spacing = 2.0 # meters
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
zb[:] += grid.node_y * 0.02 # add ramp in y direction
h = grid.add_ones('node','soil__depth') # add soil (1 m everywhere)
h[:] += 100 # soil/alluvium thickness in meters
z = grid.add_zeros('node','topographic__elevation') # add empty topography elevation
initial_roughness = np.random.rand(zb.size)/1. # make random noise to generate channels
zb[:] += initial_roughness # add random noise to bedrock
z[:] = h[:] + zb[:] # set topo elevation to be sum of soil and bedrock elevations

grid.set_fixed_value_boundaries_at_grid_edges(False, False, False, True, value = 100.,value_of='topographic__elevation')

# add channels
#channel_loc = (x/2) - 1
#z[np.where(grid.node_x == channel_loc)] -= 10
#z[np.where(grid.node_x == channel_loc+1)] -= 10
#z[np.where(grid.node_x == channel_loc-20)] -= 5
#z[np.where(grid.node_x == channel_loc+30)] -= 3

###### geomorph params ######
hstar = 0.5 # scaling parameter for weathering rate...characteristic depth
soil_decay = 0.3 # soil transport decay depth for depth diffuser
Wo = 1e-6 # initial weathering rate mm/yr [meters/yr] for exponential weatherer
#Wo = 1e-3
rhos = 1.33 * 1e3 #1.33 * 1e6 # density soil 1.33 g/cm3 --> [g/m^3]
rhor = 2.65 * 1e3 # 2.65 * 1e6 # density rock 2.65 g/cm3  --> [g/m^3]
rhos = 2.7
rhor = 2.7
kappa = 0.01  # bob says this is good value for DV alluvium. see Nash, Hanks papers. m2/yr diffusivity / transport coefficient for depth diffuser and linear diffuser
K_fs = .0009 # stream power for fastscape eroder

# parameters for EroDepo
K = 0.0003 #.0005  # erodability constant for EroDepo!
m = 0.5# m in stream power equation (exponent on Area)
n = 1. # n in stream power equation (exponent on Slope)
threshold = 0. # erosion threshold
phi = 0.0  # porosity for EroDepo
v_s = 0.001 # settling velocity for EroDepo

ideal_dt = 0.5 * grid.dx * grid.dx / kappa # calculate ideal dt based on relationship we learned about (CFL)


###### tectonics ######
fault_loc = np.int(((y*spacing) / 2) - 0)
fault_nodes = np.where(grid.node_y==fault_loc)[0]
slip_nodes = np.where(grid.node_y[grid.core_nodes] <= fault_loc)[0]


slip_interseis = 0 # magnitude not taken into account!!!
slip_seis = 1 # magnitude not taken into account!!!
slip_mag_in_grid_cells = slip_seis / grid.dx
slip_freq = np.int(400/dt)
slip_freq_high = np.int(100/dt)
slip = np.zeros(len(time))
slip[:] += slip_interseis
slip[10:-1:slip_freq_high] = slip_mag_in_grid_cells

uplift_rate = .001 # m/yr


###### plot initial conditions ######
plt.figure(figsize=(8,4))
#imshow_grid(grid,z,limits=(90,110),cmap='terrain',grid_units=['m','m'],shrink=.6)
imshow_grid(grid,z,cmap='terrain',grid_units=['m','m'],shrink=.6)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--',linewidth = 0.5)
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
depth_diffuser = DepthDependentDiffuser(grid,linear_diffusivity=kappa,soil_transport_decay_depth=soil_decay)
weatherer = ExponentialWeatherer(grid,max_soil_production_rate=Wo, soil_production_decay_depth=hstar)
flow_router = FlowRouter(grid) # optional parameters: method='D8',runoff_rate=None)
flow_accumulator = FlowAccumulator(grid,surface='topographic__elevation',flow_director='FlowDirectorD8',
                     depression_finder='DepressionFinderAndRouter') # FlowDirector D8 or Steepest
df = DepressionFinderAndRouter(grid)
fastscape_eroder = FastscapeEroder(grid,K_sp=K_fs, m_sp=m, n_sp=n, threshold_sp=threshold) #rainfall_intensity=1.)
ero_depo = ErosionDeposition(grid, K=K, phi=phi, v_s=v_s, m_sp=m, n_sp = n, 
                             sp_crit=threshold, method='simple_stream_power',
                             discharge_method=None, area_field=None, discharge_field=None)

#%% load data from topo initialization
#z[:] = np.loadtxt('ztopo',unpack=True)
z[:] = np.loadtxt('pre-fault_topo',unpack=True)

plt.figure(figsize=(8,4))
imshow_grid(grid,z,cmap='terrain',grid_units=['m','m'],shrink=.6)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--',linewidth = 0.5)
plt.title('Initial Topography')
plt.show()

#%% first make some topography - YAY WORKING - with these values: kappa = 0.01, K_sp = .0009
fs_dt = 10
iterations = 100


for i in range(iterations):
    linear_diffuser.run_one_step(fs_dt)
    #weatherer.run_one_step(fs_dt)
    #depth_diffuser.run_one_step(fs_dt)
    flow_router.run_one_step()
    #df.map_depressions()
    #flooded = np.where(df.flood_status==3)[0]
    #flow_accumulator.run_one_step()
    #fastscape_eroder.run_one_step(dt = fs_dt, flooded_nodes=flooded)
    fastscape_eroder.run_one_step(fs_dt)
    #ero_depo.run_one_step(dt)
    #zb[:] = np.minimum(zb[:],z[:])     #find where zb>z, make it z so that bedrock never goes above topo
    h[:] = z[:] - zb[:]
    z[grid.core_nodes] += uplift_rate * fs_dt

plt.figure(figsize=(8,4))
imshow_grid(grid,z,cmap='terrain',grid_units=['m','m'],shrink=.7)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--',linewidth=0.5)
plt.title('Topography after '+str(np.int((i*dt)+dt))+' years')
plt.show()

#%% save ouptut topo to load next time
f = open('ztopo','w')
np.savetxt(f,np.array(z).T) # what does .T do??
f.close()

# load data from previous run
#topo = np.loadtxt('pre-fault_topo',unpack=True)

#%%  RUN 

iterations = len(time)
#iterations = 10
plots = 2

fig = plt.figure(figsize=(8,4))
writer = animation.FFMpegWriter(fps=5) #optional: extra_args=['-vcodec', 'libx264']
writer.setup(fig, 'test.mp4', dpi=300)
imshow_grid(grid,z,cmap='terrain',grid_units=['m','m'],shrink=.7)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--',linewidth=0.5)
plt.title('Topography after 0 years')
writer.grab_frame()

for i in range(iterations):
        
    linear_diffuser.run_one_step(dt)
    #weatherer.run_one_step(dt)
    #depth_diffuser.run_one_step(dt)
    flow_router.run_one_step()
    #flow_accumulator.run_one_step()
    #fastscape_eroder.run_one_step(dt)
    ero_depo.run_one_step(dt)
    zb[:] = np.minimum(zb[:],z[:])     #find where zb>z, make it z so that bedrock never goes above topo
    h[:] = z[:] - zb[:]
    z[:] += uplift_rate * dt

    if slip[i] > 0:
        # find z values on left side, save to temp
        temp = z[grid.node_x==spacing][0:np.int((fault_loc/spacing)+1)] # this works
        #temp = z[np.where(np.logical_and(grid.node_x==1, grid.node_y<fault_loc+1))] # not tested
      
        # slip the slip nodes
        z[slip_nodes] = z[grid.neighbors_at_node[slip_nodes,0]] 
        
        # assign z values stored in temp to right side
        wanted = np.where(np.logical_and(grid.node_x==((x*spacing)-spacing), grid.node_y<fault_loc+1))
        z[wanted] = temp

    if i % plots == 0:
        print('iteration '+str(i))
        plt.clf()
        imshow_grid(grid,z,cmap='terrain',grid_units=['m','m'],shrink=.7)
        plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--',linewidth=0.5)
        plt.title('Topography after '+str(np.int((i*dt)+dt))+' years')
        writer.grab_frame()

plt.clf()        
imshow_grid(grid,z,cmap='terrain',grid_units=['m','m'],shrink=.7)
plt.plot(grid.node_x[fault_nodes],grid.node_y[fault_nodes],'k--',linewidth=0.5)
plt.title('Topography after '+str(np.int((i*dt)+dt))+' years')
writer.grab_frame()
plt.show()

writer.finish()


#%%  other plots

### show grid for testing locations
plt.figure(figsize=(8,4))
plt.plot(grid.node_x,grid.node_y,'.') 
plt.plot(grid.node_x[np.where(grid.node_x==1)[0]],grid.node_y[np.where(grid.node_x==1)[0]],'r.')
plt.plot(grid.node_x[np.where(grid.node_y<=fault_loc)[0]],grid.node_y[np.where(grid.node_y<=fault_loc)[0]],'g.')
plt.plot(grid.node_x[np.where(grid.node_x==x-1)[0]],grid.node_y[np.where(grid.node_x==1)[0]],'k.')

plt.show()

### 3D plot of topography
#surf_plot(grid,surface='topographic__elevation',title='Topographic Elevation')
