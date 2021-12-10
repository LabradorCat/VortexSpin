import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.optimize import curve_fit

# Our customized Colormap
clist = list('bkr')
clist = ['#ff0000' if c == 'r' else c for c in clist]
clist = ['#0000ff' if c == 'b' else c for c in clist]
clist = ['#000000' if c == 'k' else c for c in clist]
cmap = ListedColormap(clist)
norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

def test_cmap(cmap):
    """
    testing and showing color map
    """
    fig, ax = plt.subplots()
    x = np.linspace(0, 10, 100)
    y = x
    c = np.linspace(-1, 1, 100)
    ax.scatter(x, y, s=50, c=c, cmap=cmap)
    plt.show()

def load_summary(file, output):
    '''
        load in statistical summary from field sweep
    '''
    assert 'ASVIStateInfo' in file  # only allow summary file input

    if '.npz' not in file:
        file = file + '.npz'
    npzfile = np.load(file, allow_pickle=True)
    parameters = npzfile['arr_0']
    fieldloops = npzfile['arr_1']
    q = npzfile['arr_2']
    mag = npzfile['arr_3']
    monopole = npzfile['arr_4']
    vortex_count = npzfile['arr_5']
    macrospin_count = npzfile['arr_6']
    summary = {
        'parameters': parameters,
        'fieldloops': fieldloops,
        'q': q,
        'mag': mag,
        'monopole': monopole,
        'vortex_count': vortex_count,
        'macrospin_count': macrospin_count
    }.get(output, None)
    return summary

def plot_applied_field(field):
    # plotting external field
    steps = np.arange(0, len(field), 1)
    fig, ax = plt.subplots()
    ax.set_xlabel('Steps')
    ax.set_ylabel('Applied Field (mT)')
    ax.set_title('Applied Field on the Lattice')
    ax.plot(steps, 1000 * field, 'k-')
    # ax.plot(int(file[file.find('counter') + 7:file.find(r'_Loop')]), H_applied,
    #          marker='o', markersize=3, markerfacecolor='red', markeredgecolor='red')
    plt.show()

def plot_vortex_macrospin_number(vortex_count, macrospin_count, fitfunc):
    '''
    Plot the time evolution of vortex numbers and macrospin numbers
    '''
    assert len(vortex_count) == len(macrospin_count)
    assert 'exp' or 'sigmoid' in fitfunc

    size = len(vortex_count)
    steps = np.arange(0, size, 1)

    def exp_decay(x, a, b, c, d):
        return a * np.exp(-b * x + c) + d

    def sigmoid(x, a, b, c, d):
        return a/(1 + np.exp(-b * x + c)) + d

    fig, ax = plt.subplots()
    if fitfunc == 'exp':
        popt1, pcov1 = curve_fit(exp_decay, steps, vortex_count, p0=[np.max(vortex_count), 1, 1, 1])
        popt2, pcov2 = curve_fit(exp_decay, steps, macrospin_count,
                                 p0=[np.max(macrospin_count) / 2, 1, np.max(macrospin_count) / 2, 1])
        ax.plot(steps, exp_decay(steps, *popt1), 'r--', label='Nv fitted')
        ax.plot(steps, exp_decay(steps, *popt2), 'b--', label='Nm fitted')

    if fitfunc == 'sigmoid':
        popt1, pcov1 = curve_fit(sigmoid, steps, vortex_count, p0=[np.max(vortex_count), 1, 1, 1])
        popt2, pcov2 = curve_fit(sigmoid, steps, macrospin_count,
                                 p0=[np.max(macrospin_count) / 2, 1, np.max(macrospin_count) / 2, 1])
        ax.plot(steps, sigmoid(steps, *popt1), 'r--', label='Nv fitted')
        ax.plot(steps, sigmoid(steps, *popt2), 'b--', label='Nm fitted')

    ax.set_title('Evolution of Vortex and Macrospin Modes')
    ax.set_xlabel('steps')
    ax.set_ylabel('number of vortices')
    ax.set_xlim(0, size + 10)
    ax.set_ylim(0, np.max(macrospin_count) + 5)
    ax.plot(steps, vortex_count, 'o', label='number of vortices (Nv)')
    ax.plot(steps, macrospin_count, 'o', label='number of macrospins (Nm)')
    ax.legend()
    ax.grid()
    plt.show()

def plot_vector_field_2D(lattice, ax):
    """
    Return a vector field according to the input lattice
    Represented elements include:
        - Macrospins: red/blue arrows
        - Vertices: colored dots with color representing vertex charges
        - Vortices: black dots
    Input parameter specifications:
        - lattice: 3D numpy.ndarray from the ASVI class
        - ax: the matplotlib axis to plot the vector field
    """
    # extracting useful values from lattice
    X = lattice[:, :, 0].flatten()
    Y = lattice[:, :, 1].flatten()
    bar_l = lattice[:, :, 7].flatten()
    bar_w = lattice[:, :, 8].flatten()
    Mx = lattice[:, :, 3].flatten()
    My = lattice[:, :, 4].flatten()
    U = Mx * bar_l  # normalise vector
    V = My * bar_l  # normalise vector
    Cv = lattice[:, :, 10].flatten()
    # sorting out colors and thicknesses
    line_w = 4 * (bar_w > 100e-9) + 1
    line_rbg = np.arctan(Mx + My)
    color = cmap(norm(line_rbg))
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, pivot='mid', zorder=1,
              linewidths=line_w, color=color, edgecolors=color)
    ax.scatter(X, Y, s=50, c=Cv, cmap='gist_rainbow', marker='o', zorder=2, vmax=1, vmin=-1)
    ax.set_xlabel('Lx (m)')
    ax.set_ylabel('Ly (m)')
    ax.ticklabel_format(style='sci', scilimits=(0, 0))
    ax.use_sticky_edges = False
    return ax

if __name__ == '__main__':
    file = 'D:\ASI_MSci_Project\ASVI_Simulation_Output\ASVIStateInfo_Hmax0.030_steps100_Angle45_n2_Loops0.npz'
    vc = load_summary(file, output='vortex_count')
    mc = load_summary(file, output='macrospin_count')
    fd = load_summary(file, output='fieldloops')
    plot_applied_field(fd)
    plot_vortex_macrospin_number(vc, mc, 'sigmoid')
