
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
# Material Parameters

size = 2  ## Dimension of array

Hc = 0.062  # Coercive Field (T)
Hc_std = 5  # Stanard deviation in the coercive field (as a percentage)
bar_length = 400e-9  # Bar length in m
vertex_gap = 100e-9  # Vertex gap in m
bar_thickness = 20.5e-9  # Bar thickness in m
bar_width = 80e-9  # Bar width in m
magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)
field_angle = 45.  # Angle at which the field will be applied in degrees
field_max = 0.95 * Hc / np.cos(field_angle / 180 * np.pi)  # Maximum field to by applied at field angle measured in Telsa
steps = 5  # Number of steps between the minimum value of the coercive field
magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)

#-----------------------------------------------------------------------------------------------------------------------
# Lattice Parameters
# Define the size of the lattice

Hsteps = 5  # Number of steps between the minimum value of the coercive field
            # and the maxium field specified above. Total number of steps in a
            # minor loop is = 4*(steps+1)
neighbours = 4  # The radius of neighbouring spins that are included in the local field calculation
loops = 6  # The number of minor field loops to be done

#-----------------------------------------------------------------------------------------------------------------------
# Generate ASCI Class model

lattice = asvi.ASVI(size, size, bar_length=bar_length,
                      vertex_gap=vertex_gap, bar_thickness=bar_thickness,
                      bar_width=bar_width, magnetisation=magnetisation)
lattice.square(Hc, Hc_std / 100)  # Specify whether it is a square or kagome lattice

#-----------------------------------------------------------------------------------------------------------------------
# Running Simulation and output results

output_folder_name = 'ASVI_Simulation_Test'
# Simulation results export to parent directory

folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, output_folder_name))
if os.path.exists(folder) == False:
    os.mkdir(folder)

lattice.fieldSweepAdaptive(Hmax=0.07, steps=10, Htheta=46, n=10, loops=1, folder=folder, q1=False)
lattice.fieldSweepAnimation(folder, name='Lattice_counter')


