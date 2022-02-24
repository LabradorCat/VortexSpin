import shutil
import os
from asvi import ASVI

#-----------------------------------------------------------------------------------------------------------------------
# Material & Lattice Parameters
# Define the size of the lattice and material properties
size = 5  ## Dimension of array
Hc_thin = 0.029  # Coercive Field (T)
Hc_thick = 0.01625
Hc_Vortex = 0.026
Hc_std = 1.5  # Stanard deviation in the coercive field (as a percentage)
vertex_gap = 100e-9  # Vertex gap in m
bar_length = 600e-9  # Bar length in m
bar_thickness = 20e-9  # Bar thickness in m
thin_bar_width = 125e-9  # Bar width in m
thick_bar_width = 200e-9
magnetisation = 800e3  # Saturation magnetisation of material in A/m (permalloy is 80e3)
field_angle = 45.  # Angle at which the field will be applied in degrees
field_max = 0.023  # Maximum field to by applied at field angle measured in Telsa

#-----------------------------------------------------------------------------------------------------------------------
# Simulation Parameters
Field = 'Adaptive'  # Type of Field used to sweep the lattice
Hsteps = 20         # Number of steps between the minimum value of the coercive field
                    # and the maxium field specified above. Total number of steps in a
                    # minor loop is = (2*steps)
neighbours = 2      # The radius of neighbouring spins that are included in the local field calculation
loops = 10         # The number of minor field loops to be done

#-----------------------------------------------------------------------------------------------------------------------
# Running Simulation and output results
output_folder_name = 'ASVI_Simulation_Output' # Simulation results export to 'output_folder_name' in the parent directory
fps = 10    # Animation fps
# Select what to perform in this run
Simulate = True
Animate = True

if __name__ == '__main__':
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, output_folder_name))
    # Generate ASVI Class model
    lattice = ASVI(size, size, vertex_gap=vertex_gap)
    lattice.square_staircase(Hc_thin, Hc_thick, Hc_Vortex, Hc_std / 100, magnetisation,
                             bar_length, thin_bar_width, thick_bar_width, bar_thickness)

    if os.path.exists(folder) == False:
        os.mkdir(folder)

    if Simulate:
        shutil.rmtree(folder)
        lattice.fieldSweep(fieldType=Field, Hmax=field_max, steps=Hsteps, Htheta=field_angle,
                           n=neighbours, loops=loops, folder=folder, FMR=True, FMR_field=-0.0012)

    if Animate:
        lattice.fieldSweepAnimation(folder, fps=fps)
