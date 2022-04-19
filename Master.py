import shutil
import os
from asvi import ASVI
from plotting import load_summary, FMR_specturm
# -----------------------------------------------------------------------------------------------------------------------
# Material & Lattice Parameters
# Define the size of the lattice and material properties
size = 100              # Dimension of array
Hc_thin = 0.030             # Coercive Fields (T)
Hc_thick = 0.015
Hc_vortex = 0.022
Hc_std = 0.015               # Standard deviation in the coercive field (percentage)
Hc_v_std = 0.03             # Standard deviation in the vortex coercive field (percentage)
vertex_gap = 100e-9         # Vertex gap in m
bar_length = 600e-9         # Bar length in m
bar_thickness = 20e-9       # Bar thickness in m
thin_bar_width = 125e-9     # Bar width in m
thick_bar_width = 200e-9
magnetisation = 750e3       # Saturation magnetisation of material in A/m (permalloy is 80e3)
field_angle = 45.           # Angle at which the field will be applied in degrees
field_max = 0.0250          # Maximum field to by applied at field angle measured in Telsa
field_min = 0.0200          # Minimum field to by applied at field angle measured in Telsa
# -----------------------------------------------------------------------------------------------------------------------
# Simulation Parameters
Field = 'Sine_Train'          # Type of Field used to sweep the lattice
InterType = 'dumbbell'      # Type of interaction (dumbbell or dipole)
PeriodicBC = False          # Apply periodic boundary condition
Hsteps = 30                  # Number of steps between the minimum value of the coercive field
                            # and the maxium field specified above. Total number of steps in a
                            # minor loop is = (2*steps)
neighbours = 2              # The radius of neighbouring spins that are included in the local field calculation
loops = 20                   # The number of minor field loops to be done

# -----------------------------------------------------------------------------------------------------------------------
# FMR Parameters
FMR = True
FMR_folder_name = 'FMR_Simulation_Output'       # export folder for FMR data
FMR_file_name = f'{Field}_size{size}_{int(field_min*1000)}to{int(field_max*1000)}mT_S{Hsteps}L{loops}.xlsx'

FMR_step = 2
FMR_field = None
freq_min = 2                    # minimum frequency in FMR spectrum (GHz)
freq_max = 11                   # maximum frequency in FMR spectrum (GHz)
IQ_bins = 396                   # number of frequency bins
bandwidth = 0.1                 # bandwidth for KDE smoothing (Low-pass-filter-like mechanism)

# -----------------------------------------------------------------------------------------------------------------------
# Running Simulation and output results
output_folder_name = 'ASVI_Simulation_Output'   # export folder for simulation data
fps = 10                                        # Animation fps
animation_size = (40, 40)                       # Animation figure size
# Select what to perform in this run
Simulate = True
Animate = True

if __name__ == '__main__':
    folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, output_folder_name))
    FMR_folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir, FMR_folder_name))
    # Generate ASVI Class model
    lattice = ASVI(size, size, vertex_gap, InterType, PeriodicBC)
    lattice.square_staircase(Hc_thin, Hc_thick, Hc_vortex, Hc_std, Hc_v_std, magnetisation,
                             bar_length, thin_bar_width, thick_bar_width, bar_thickness)

    if not os.path.exists(folder):
        os.mkdir(folder)

    if Simulate:
        shutil.rmtree(folder)
        lattice.fieldSelect(fieldType=Field, steps=Hsteps, Hmax=field_max, Hmin=field_min, Htheta=field_angle)
        lattice.fieldSweep(n=neighbours, loops=loops, folder=folder, FMR=FMR, FMR_step=FMR_step, FMR_field=FMR_field)
        if FMR:
            if not os.path.exists(FMR_folder):
                os.mkdir(FMR_folder)
            # produce FMR data sheet
            FMR_output_path = os.path.join(FMR_folder, FMR_file_name)
            FMR_f = load_summary(folder, output='FMR_frequency')
            FMR_specturm(FMR_f, FMR_output_path, fmin=freq_min, fmax=freq_min, bins=IQ_bins, bandwidth=bandwidth)

    if Animate:
        lattice.fieldSweepAnimation(folder, figsize=animation_size, fps=fps)
