# import Plasimage_recogntion
import os
import rpmClass_Vortex as rpm
from importlib import *

import numpy as np
import os
import matplotlib.pyplot as plt
from skimage import data, feature, exposure
from PIL import Image
import numpy as np
import argparse
#import cv2
import math
#from shapely import geometry
from matplotlib import pyplot as plt
from scipy import spatial
import itertools
import matplotlib.cm as cm
import matplotlib.colors as cl
from functools import partial

reload(rpm)


def distance_squared(x, y):
    return (x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2


# imageim = r"\\icnas2.cc.ic.ac.uk\kjs18\GitHub\RPMv3\Latticeimages\10 x10-GS.png" #Path to MFM image
# latticeim = r"\\icnas2.cc.ic.ac.uk\kjs18\GitHub\RPMv3\Latticeimages\10 x10-GS.png" #Path to lattice image
# folder = r'C:\Users\kjs18\Documents\RPM\RPM Code\Data\Defect_Simualtion\Tmonopole\Hc_std5\field_angle45\maxH0.97\attempt2'	#The folder for the files to be saved in.
size = 2  ## Dimension of array
fname = 'GS'  ## Save Mx and My
# file = r"\\icnas2.cc.ic.ac.uk\kjs18\GitHub\RPMv3\defectnpz\10x10GS.npz"

## Run this for recognition
# latticeim = r"D:\Samples_exp\Alex_fingerprint_mfm\jcg-rpm-vtx-mfm-440-4wide.000mfm.png"
# imageim = r"D:\Samples_exp\Alex_fingerprint_mfm\jcg-rpm-vtx-mfm-440-4wide.000mfm.png"
# Mx,My = square_symplify.square(imageim, latticeim, size,fname)


## Use this to load Mx and My files
# Mx = r''
# My = r''


# Material Parameters

Hc = 0.062  # Coercive Field (T)
Hc_std = 5  # Stanard deviation in the coercive field (as a percentage)
bar_length = 400e-9  # Bar length in m
vertex_gap = 100e-9  # Vertex gap in m
bar_thickness = 20.5e-9  # Bar thickness in m
bar_width = 80e-9  # Bar width in m
magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)

field_angle = 45.  # Angle at which the field will be applied in degrees
field_max = 0.95 * Hc / np.cos(
    field_angle / 180 * np.pi)  # Maximum field to by applied at field angle measured in Telsa
steps = 5  # Number of steps between the minimum value of the coercive field

magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)

# Lattice Parameters
# Define the size of the lattice

Hsteps = 5  # Number of steps between the minimum value of the coercive field
# and the maxium field specified above. Total number of steps in a
# minor loop is = 4*(steps+1)
neighbours = 4  # The radius of neighbouring spins that are included in the local
# field calculation
loops = 6  # The number of minor field loops to be done

lattice = rpm.ASI_RPM(size, size, bar_length=bar_length,
                      vertex_gap=vertex_gap, bar_thickness=bar_thickness,
                      bar_width=bar_width, magnetisation=magnetisation)
lattice.square(Hc, Hc_std / 100)  # Specify whether it is a square or kagome lattice
'''
lattice.load(file)
lattice.coerciveVertex(Hc)
plt.show()
'''

## Function to load in the Mx and My files
# lattice.mfmLoad(Mx,My)
# lattice.vertexTypeMap()
#lattice.graphCharge()
#plt.show()

folder = os.path.join(os.getcwd(), 'test')
if os.path.exists(folder) == False:
    os.mkdir(folder)
lattice.fieldSweepAdaptive(Hmax=0.07, steps=10, Htheta=46, n=10, loops=1, folder=folder, q1=False)

lattice.fieldSweepAnimation2(folder, name='Lattice_counter')
'''
flipme = [(1,2), (2,5), (3,4), (4,1), (4,3), (5,4)]

for spins in flipme:
	lattice.flipSpin(spins[0]+6,spins[1]+6)


for x in range(18):
	for y in range(18):
		lattice.flipSpin(x,y)

lattice.vertexTypeMa2()
plt.show()

'''

