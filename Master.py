
import asvi_models as asvi
import shutil
import matplotlib.pyplot as plt
import os
import numpy as np
from importlib import *
from time import time, sleep
from plotting import plot_vortex_macrospin_number, load_summary

reload(asvi)
#-----------------------------------------------------------------------------------------------------------------------
# Material & Lattice Parameters
# Define the size of the lattice and material properties
size = 2  ## Dimension of array
Hc_thin = 0.025  # Coercive Field (T)
Hc_thick = 0.015
Hc_Vortex = 0.020
Hc_std = 2  # Stanard deviation in the coercive field (as a percentage)
bar_length = 400e-9  # Bar length in m
vertex_gap = 100e-9  # Vertex gap in m
bar_thickness = 20.5e-9  # Bar thickness in m
thin_bar_width = 100e-9  # Bar width in m
thick_bar_width = 200e-9
magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)
field_angle = 45.  # Angle at which the field will be applied in degrees
field_max = 0.0192  # Maximum field to by applied at field angle measured in Telsa
magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)

#-----------------------------------------------------------------------------------------------------------------------
# Simulation Parameters
Field = 'Sine'  # Type of Field used to sweep the lattice
Hsteps = 20         # Number of steps between the minimum value of the coercive field
                    # and the maxium field specified above. Total number of steps in a
                    # minor loop is = (2*steps)
neighbours = 2      # The radius of neighbouring spins that are included in the local field calculation
loops = 10          # The number of minor field loops to be done

#-----------------------------------------------------------------------------------------------------------------------
# Benchmarking Functions
def test_sim_speed(maxsize, Hsteps, neighbours, loops):
    size = 2
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'ASVI_test_cache'))
    time_array = []
    while size <= maxsize:
        test_asvi = asvi.ASVI(size, size, bar_length=bar_length, bar_width=thin_bar_width, bar_thickness=bar_thickness,
                            vertex_gap=vertex_gap, magnetisation=magnetisation)
        test_asvi.square_staircase(Hc_thin, Hc_thick, Hc_std / 100, thick_bar_width)
        t1 = time()
        test_asvi.fieldSweep(fieldType='Adaptive', Hmax=field_max, steps=Hsteps, Htheta=field_angle,
                           n=neighbours, loops=loops, folder=folder, q1=False)
        t2 = time()
        time_array.append(t2 - t1)
        shutil.rmtree(folder)
        sleep(1)
        size += 1

    plt.figure('Simulation Speed Benchmark')
    plt.title('Simulation Time Test')
    plt.xlabel('Size')
    plt.xlim(0, maxsize + 1)
    plt.ylabel('Simulation Time (sec)')
    plt.ylim(0, np.max(time_array) + 5)
    step = np.arange(2, maxsize + 1, 1, dtype = int)
    plt.plot(step, time_array, 'o')
    plt.grid()
    plt.show()


def test_vortex_exp(maxsize = 20, step = 5):
    size = 5
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'ASVI_test_cache'))
    fig, ax = plt.subplots()
    while size <= maxsize:
        test_asvi = asvi.ASVI(size, size, bar_length=bar_length, bar_width=thin_bar_width, bar_thickness=bar_thickness,
                            vertex_gap=vertex_gap, magnetisation=magnetisation)
        test_asvi.square_staircase(Hc_thin, Hc_thick, Hc_std / 100, thick_bar_width)
        test_asvi.fieldSweep(fieldType='Adaptive', Hmax=field_max, steps=Hsteps, Htheta=field_angle,
                             n=neighbours, loops=loops, folder=folder, q1=False)
        vc = load_summary(folder, output='vortex_count')
        mc = load_summary(folder, output='macrospin_count')
        plot_vortex_macrospin_number(vc, mc, 'exp', ax)
        shutil.rmtree(folder)
        sleep(1)
        size += step
    plt.show()


def test_vortex_sig(maxsize = 20, step = 5):
    size = 5
    field_max = 0.02
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, 'ASVI_test_cache'))
    fig, ax = plt.subplots()
    while size <= maxsize:
        test_asvi = asvi.ASVI(size, size, bar_length=bar_length, bar_width=thin_bar_width, bar_thickness=bar_thickness,
                            vertex_gap=vertex_gap, magnetisation=magnetisation)
        test_asvi.square_staircase_vortex(Hc_thin, Hc_thick, Hc_std / 100, thick_bar_width)
        test_asvi.fieldSweep(fieldType='Linear', Hmax=field_max, steps=200, Htheta=field_angle,
                             n=neighbours, loops=0, folder=folder, q1=False)
        vc = load_summary(folder, output='vortex_count')
        mc = load_summary(folder, output='macrospin_count')
        plot_vortex_macrospin_number(vc, mc, 'sigmoid', ax)
        shutil.rmtree(folder)
        sleep(1)
        size += step
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# Running Simulation and output results
output_folder_name = 'ASVI_Simulation_Output' # Simulation results export to 'output_folder_name' in the parent directory
fps = 10    # Animation fps
# Select what to perform in this run
Simulate = True
Animate = False
Test = False

if __name__ == '__main__':
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, output_folder_name))
    # Generate ASVI Class model
    lattice = asvi.ASVI(size, size, bar_length=bar_length, bar_width=thin_bar_width, bar_thickness=bar_thickness,
                        vertex_gap=vertex_gap, magnetisation=magnetisation)
    lattice.square(Hc_thin, Hc_std / 100)
    if os.path.exists(folder) == False:
        os.mkdir(folder)
    if Simulate:
        shutil.rmtree(folder)
        lattice.fieldSweep(fieldType = Field, Hmax = field_max, steps = Hsteps, Htheta = field_angle,
                           n = neighbours, loops = loops, folder = folder, q1 = False)
    if Animate:
        lattice.fieldSweepAnimation(folder, fps = fps)
    if Test:
        #test_sim_speed(maxsize=3, Hsteps=20, neighbours=2, loops=5)
        #test_vortex_exp(20, 5)
        test_vortex_sig(10, 5)