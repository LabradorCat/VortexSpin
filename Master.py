
import ASVI_Class as asvi
from importlib import *
import os
import numpy as np
# import Plasimage_recogntion
#import matplotlib.pyplot as plt
#from skimage import data, feature, exposure
#from PIL import Image
#import argparse
#import cv2
#import math
#from shapely import geometry
#from matplotlib import pyplot as plt
#from scipy import spatial
#import itertools
#import matplotlib.cm as cm
#import matplotlib.colors as cl
#from functools import partial

reload(asvi)
#-----------------------------------------------------------------------------------------------------------------------
# Material & Lattice Parameters
# Define the size of the lattice and material properties
size = 5  ## Dimension of array

Hc_thin = 0.025  # Coercive Field (T)
Hc_thick = 0.018
Hc_Vortex = 0.020

Hc_std = 2  # Stanard deviation in the coercive field (as a percentage)
bar_length = 400e-9  # Bar length in m
vertex_gap = 100e-9  # Vertex gap in m
bar_thickness = 20.5e-9  # Bar thickness in m

thin_bar_width = 100e-9  # Bar width in m
thick_bar_width = 200e-9

magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)
field_angle = 45.  # Angle at which the field will be applied in degrees
field_max = 0.0225  # Maximum field to by applied at field angle measured in Telsa
magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)

#-----------------------------------------------------------------------------------------------------------------------
# Simulation Parameters
Field = 'Adaptive'  # Type of Field used to sweep the lattice
Hsteps = 10         # Number of steps between the minimum value of the coercive field
                    # and the maxium field specified above. Total number of steps in a
                    # minor loop is = (2*steps)
neighbours = 2      # The radius of neighbouring spins that are included in the local field calculation
loops = 1        # The number of minor field loops to be done

#-----------------------------------------------------------------------------------------------------------------------
# Generate ASCI Class model
lattice = asvi.ASVI(size, size, bar_length = bar_length, bar_width = thin_bar_width, bar_thickness = bar_thickness,
                    vertex_gap=vertex_gap, magnetisation=magnetisation)
lattice.square_staircase(Hc_thin, Hc_thick, Hc_std / 100, thick_bar_width)  # Specify whether it is a square or kagome lattice

#-----------------------------------------------------------------------------------------------------------------------
# Running Simulation and output results
output_folder_name = 'ASVI_Simulation_Output' # Simulation results export to 'output_folder_name' in the parent directory
fps = 10    # Animation fps

# Select what to perform in this run
Simulate = True
Animate = True
Show_statistics = True

if __name__ == '__main__':
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, output_folder_name))
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    if Simulate:
        lattice.fieldSweep(fieldType = Field, Hmax = field_max, steps = Hsteps, Htheta = field_angle,
                           n = neighbours, loops = loops, folder = folder, q1 = False)
    if Animate:
        lattice.fieldSweepAnimation(folder, fps = fps)
