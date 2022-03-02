import shutil
import os
from asvi import ASVI

# -----------------------------------------------------------------------------------------------------------------------
# Material & Lattice Parameters
# Define the size of the lattice and material properties
size = 10                # Dimension of array
Hc_thin = 0.029             # Coercive Fields (T)
Hc_thick = 0.01625
Hc_Vortex = 0.026
Hc_std = 0.015              # Stanard deviation in the coercive field (as a percentage)
vertex_gap = 100e-9         # Vertex gap in m
bar_length = 600e-9         # Bar length in m
bar_thickness = 20e-9       # Bar thickness in m
thin_bar_width = 125e-9     # Bar width in m
thick_bar_width = 200e-9
magnetisation = 800e3       # Saturation magnetisation of material in A/m (permalloy is 80e3)
field_angle = 45.           # Angle at which the field will be applied in degrees
field_max = 0.0235          # Maximum field to by applied at field angle measured in Telsa
field_min = 0.0180          # Minimum field to by applied at field angle measured in Telsa
# -----------------------------------------------------------------------------------------------------------------------
# Simulation Parameters
Field = 'MackeyGlass'        # Type of Field used to sweep the lattice
InterType = 'dumbbell'      # Type of interaction (dumbbell or dipole)
PeriodicBC = False        # Apply periodic boundary condition
Hsteps = 300                 # Number of steps between the minimum value of the coercive field
                            # and the maxium field specified above. Total number of steps in a
                            # minor loop is = (2*steps)
neighbours = 2              # The radius of neighbouring spins that are included in the local field calculation
loops = 0                   # The number of minor field loops to be done

# -----------------------------------------------------------------------------------------------------------------------
# FMR Parameters
FMR = True
FMR_step = 2
FMR_field = None

# -----------------------------------------------------------------------------------------------------------------------
# Running Simulation and output results
output_folder_name = 'ASVI_Simulation_Output'   # export to 'output_folder_name' in the parent directory
fps = 10                                        # Animation fps
animation_size = (10, 10)                        # Animation figure size
# Select what to perform in this run
Simulate = True
Animate = True

if __name__ == '__main__':
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, output_folder_name))
    # Generate ASVI Class model
    lattice = ASVI(size, size, vertex_gap, InterType, PeriodicBC)
    lattice.square_staircase(Hc_thin, Hc_thick, Hc_Vortex, Hc_std, magnetisation,
                             bar_length, thin_bar_width, thick_bar_width, bar_thickness)

    if not os.path.exists(folder):
        os.mkdir(folder)

    if Simulate:
        shutil.rmtree(folder)
        lattice.fieldSweep(fieldType=Field, steps=Hsteps, Hmax=field_max, Hmin=field_min, Htheta=field_angle,
                           n=neighbours, loops=loops, folder=folder, FMR=FMR, FMR_step=FMR_step, FMR_field=FMR_field)

    if Animate:
        lattice.fieldSweepAnimation(folder, figsize=animation_size, fps=fps)
