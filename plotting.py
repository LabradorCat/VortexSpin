import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

# Customized colormap for nanobars
clist1 = list('bkr')
clist1 = ['#ff0000' if c == 'r' else c for c in clist1]
clist1 = ['#0000ff' if c == 'b' else c for c in clist1]
clist1 = ['#000000' if c == 'k' else c for c in clist1]
mycmap = ListedColormap(clist1)
mynorm = BoundaryNorm([-1, -0.5, 0.5, 1], mycmap.N)


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


def load_summary(folder, output):
    '''
        load in statistical summary from field sweep
    '''
    summary = None
    for root, dirs, files in os.walk(folder):
        sum_files = list(filter(lambda x: 'ASVIStateInfo' in x, files))
        for file in sum_files:
            if '.npz' not in file:
                file = file + '.npz'
            npzfile = np.load(os.path.join(root, file), allow_pickle=True)
            parameters = npzfile['arr_0']
            fieldloops = npzfile['arr_1']
            q = npzfile['arr_2']
            mag = npzfile['arr_3']
            monopole = npzfile['arr_4']
            vortex_count = npzfile['arr_5']
            macrospin_count = npzfile['arr_6']
            frequency = npzfile['arr_7']
            summary = {
                'parameters': np.asarray(parameters),
                'fieldloops': np.asarray(fieldloops),
                'q': np.asarray(q),
                'mag': np.asarray(mag),
                'monopole': np.asarray(monopole),
                'vortex_count': np.asarray(vortex_count),
                'macrospin_count': np.asarray(macrospin_count),
                'FMR_frequency': np.asarray(frequency)
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


def plot_vortex_macrospin_number(vortex_count, macrospin_count, fitfunc, ax=None):
    '''
    Plot the time evolution of vortex numbers and macrospin numbers
    '''
    assert len(vortex_count) == len(macrospin_count)
    assert 'exp' or 'sigmoid' in fitfunc

    if ax is None:
        fig, ax = plt.subplots()

    size = len(vortex_count)
    steps = np.arange(0, size, 1)

    def exp_decay(x, a, b, c):
        return a * np.exp(-b * x) + c

    def sigmoid(x, a, b, c, d):
        return (a / (1 + np.exp(-b * x + c))) + d

    if fitfunc == 'exp':
        popt1, pcov1 = curve_fit(exp_decay, steps, vortex_count,
                                 p0=[-np.max(vortex_count), 1, np.max(vortex_count)])
        popt2, pcov2 = curve_fit(exp_decay, steps, macrospin_count,
                                 p0=[np.max(macrospin_count) / 2, 1, np.max(macrospin_count) / 2])
        ax.plot(steps, exp_decay(steps, *popt1), 'r--', zorder=2)
        ax.plot(steps, exp_decay(steps, *popt2), 'b--', zorder=2)

    if fitfunc == 'sigmoid':
        popt1, pcov1 = curve_fit(sigmoid, steps, vortex_count,
                                 p0=[np.min(vortex_count)-np.max(vortex_count), 1, 5, np.max(vortex_count)])
        popt2, pcov2 = curve_fit(sigmoid, steps, macrospin_count,
                                 p0=[np.max(macrospin_count)-np.min(macrospin_count), 1, 5, np.min(macrospin_count)])
        ax.plot(steps, sigmoid(steps, *popt1), 'r--', zorder=2)
        ax.plot(steps, sigmoid(steps, *popt2), 'b--', zorder=2)

    ax.set_title('Evolution of Vortex and Macrospin Modes')
    ax.set_xlabel('steps')
    ax.set_ylabel('number of vortices')
    ax.set_xlim(0, size + 10)
    ax.set_ylim(0, np.max(macrospin_count) + 5)
    ax.plot(steps, vortex_count, 'o', label='number of vortices (Nv)', zorder=1)
    ax.plot(steps, macrospin_count, 'o', label='number of macrospins (Nm)', zorder=1)
    ax.legend()
    ax.grid()


def plot_vector_field_2D(asvi, fig, ax, color=None):
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
    pos = asvi.pos_matrix()
    mag = asvi.mag_matrix()
    vc = asvi.vc_matrix()
    X = pos[:, :, 0].flatten()
    Y = pos[:, :, 1].flatten()
    Mx = mag[:, :, 0].flatten()
    My = mag[:, :, 1].flatten()
    Cv = vc.flatten()
    bar_l = asvi.get_attribute_matrix('bar_l').flatten()
    bar_w = asvi.get_attribute_matrix('bar_w').flatten()
    bar_w[np.isnan(bar_w)] = 0
    U = Mx * bar_l  # normalise vector
    V = My * bar_l  # normalise vector
    # sorting out colors and thicknesses
    line_w = 4 * (bar_w > 150e-9) + 1
    if color is None:
        line_rbg = np.arctan(Mx + My)
        color = mycmap(mynorm(line_rbg))
    # plotting vector field
    ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, pivot='mid', zorder=1,
              linewidths=line_w, color=color, edgecolors=color)
    ax.scatter(X, Y, s=50, c=Cv, cmap='gist_rainbow', marker='o', zorder=2, vmax=1, vmin=-1)
    ax.set_xlabel('Lx (m)')
    ax.set_ylabel('Ly (m)')
    ax.ticklabel_format(style='sci', scilimits=(0, 0))
    ax.use_sticky_edges = False
    return ax


def plot_FMR(FMR_frequency):

    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    ax1, ax2 = axes[0], axes[1]

    def kde_sklearn(x, x_grid, bandwidth=0.2, **kwargs):
        """Kernel Density Estimation with Scikit-learn"""
        kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
        kde_skl.fit(x[:, np.newaxis])
        # score_samples() returns the log-likelihood of the samples
        log_pdf = kde_skl.score_samples(x_grid[:, np.newaxis])
        return np.exp(log_pdf)
    # Setting color code
    num_plots = len(FMR_frequency)
    offset = 2
    f_max = np.max(FMR_frequency) + offset
    f_min = np.min(FMR_frequency) - offset
    f_arr = np.arange(f_min, f_max, 0.01)
    # Plotting
    fig.suptitle('FMR Spectrum')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Occurrence')
    ax1.set_xlim(f_min, f_max)
    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Probability')

    loop = 0
    for i in range(0, num_plots):
        freq = FMR_frequency[i]
        f_pdf = kde_sklearn(freq, f_arr, bandwidth=0.2)
        if i == 0:
            ax1.hist(freq, bins=10, histtype='step', label='Initial', color = 'blue', linewidth=2, alpha=1)
            ax2.plot(f_arr, f_pdf, label='Initial', color = 'blue', linewidth=2, alpha=1)
        elif i == num_plots - 1:
            ax1.hist(freq, bins=10, histtype='step', label='Final', color='red', linewidth=2, alpha=1)
            ax2.plot(f_arr, f_pdf, label='Final', color='red', linewidth=2, alpha=1)
        elif i % 5 == 0:
            ax1.hist(freq, bins=10, histtype='step', label=f'loop {loop}', linewidth=3, alpha=0.5)
            ax2.plot(f_arr, f_pdf, label=f'loop {loop}', linewidth=3, alpha=0.5)
        loop += 1
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')


def FMR_heatmap(type=0, field=0, display_FMR_heatmap=False):
    if display_FMR_heatmap:
        uH = np.linspace(-10, 10, 100)
        plt.figure('FMR_heatmap', figsize=(3,3))
        plt.xlabel('uH(mT)')
        plt.xlim(-10, 10)
        plt.ylabel('frequency (GHz)')
        plt.ylim(2, 11)
        plt.plot(uH, FMR_heatmap(0, uH))
        plt.plot(uH, FMR_heatmap(1, uH))
        plt.plot(uH, FMR_heatmap(2, uH))
        plt.grid()
        plt.show()
    else:
        if type == 0:  # thin bar
            return -0.035 * field + 8.7
        if type == 1:  # thick bar
            return -0.035 * field + 6.9
        if type == 2:  # vortex
            return -1.86 * np.tanh(0.2 * field) + 4.5


if __name__ == '__main__':
    folder = 'E:\ASI_MSci_Project\ASVI_Simulation_Output'
    vc = load_summary(folder, output='vortex_count')
    mc = load_summary(folder, output='macrospin_count')
    # fd = load_summary(folder, output='fieldloops')
    FMR_f = load_summary(folder, output='FMR_frequency')
    # plot_applied_field(fd)
    #plot_vortex_macrospin_number(vc, mc, 'exp')
    #FMR_heatmap(display_FMR_heatmap=True)
    plot_FMR(FMR_f)